import operator
import time
from contextlib import contextmanager
from typing import Annotated, Optional, TypedDict

from langfuse import get_client
from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz

from app import canonical_mapper, context_builder, llm, pipeline, prompts, understanding
from app.config import settings
from app.models import TraceEntry
from app.tasks import TaskSpec, TaskType


@contextmanager
def _node_span(node_name: str):
    """Groups a node's underlying LLM call(s) under one named step in the
    Langfuse trace waterfall — the auto-instrumented generation (with real
    token usage/cost/latency) nests under this span."""
    with get_client().start_as_current_observation(name=node_name, as_type="span") as span:
        yield span


class GraphState(TypedDict):
    session_id: str
    project_id: str
    message: str
    history: str
    # Rooms currently deprioritized by a decline — see classify_intent_node's
    # decline-detection step. A room only ever moves INTO this list; there's
    # no explicit "unskip" (see app.question_engine.find_knowledge_gaps'
    # auto-advance).
    skipped_rooms: list[str]
    # {"canonical_paths", "room_id"} for the field(s) the LAST turn's
    # question was about — read only by classify_intent_node (feeds
    # decline-detection via room_id). Written by app.chat from the previous
    # turn's session.current_field.
    current_field: Optional[dict]
    # Which room a batched question/write is currently focused on — see
    # ChatSession.active_room_id's docstring (app/models.py). Written by
    # classify_intent_node (from the turn's resolved tasks), corrected by
    # app.pipeline.run_pipeline for a brand-new room its own write just
    # created. Never used for write-target grounding — that stays
    # exclusively task.connection/room_hint.
    active_room_id: Optional[str]
    intent: list[str]
    tasks: list[TaskSpec]
    # Set by classify_intent_node when classify_operations returned a write
    # task (EDIT_CONTEXT/DELETE_CONTEXT/RETRIEVE_CONTEXT) with connection=None
    # and/or confusion=True — one clarifying question per unresolved op,
    # batched together (see the classifier-connection plan, generalized to
    # content forks). Shaped {"tasks": [TaskSpec.model_dump(), ...],
    # "questions": [{"op_id", "text", "question", "resolution_type",
    # "options"}, ...]} — stashes the WHOLE task list
    # (resolved and unresolved alike) so the resume turn never needs to
    # re-classify.
    pending_operation_questions: Optional[dict]
    # {op_id: chosen_value} from the client, answering a prior turn's
    # pending_operation_questions — read by classify_intent_node's resume
    # branch, never set by any graph node itself.
    operation_answers: Optional[dict[str, str]]
    # {op_id: [room_id, ...]} from the client — set ONLY when the chosen
    # answer for that op_id was a bundled multi-room option (see
    # app.llm.Option.room_ids / the RESOLUTION_AGENT multi-room rule).
    # operation_answers[op_id] still carries that option's
    # display label for the transcript; this is the parallel, structured
    # grounding data classify_intent_node's resume branch uses to fan the
    # ONE op out into one write task per room id instead of setting a single
    # connection. Never set by any graph node itself.
    operation_room_selections: Optional[dict[str, list[str]]]
    # Serialized app.question_engine.KnowledgeGapBatch for the still-open
    # field(s), if any — written by app.pipeline.run_pipeline every turn.
    pending_gap: Optional[dict]
    # DIRECT_ANSWER's streamed reply text, if this turn had one.
    answer: str
    # The pipeline's single join-step summary output (see
    # app.pipeline.run_pipeline / app.llm.generate_turn_summary) — each is
    # None when that piece didn't apply this turn.
    database_summary: Optional[str]
    context_summary: Optional[str]
    changes_summary: Optional[str]
    # The next question to ask, or a closing/completion line — see
    # `is_question`. Named "next_message", NOT "message": GraphState.message
    # is already this turn's own user input text.
    next_message: Optional[str]
    is_question: bool
    # True once nothing is left open project-wide (question_engine.
    # find_knowledge_gaps returned None) — app.pipeline.run_pipeline also
    # materializes the completion ProjectContext snapshot in that case.
    complete: bool
    trace: Annotated[list[TraceEntry], operator.add]


# Phrases that count as an explicit decline of the currently pending question.
# Deliberately narrow and deterministic (no extra LLM call, consistent with
# this codebase's latency-consciousness around model calls — see the
# model_extraction/Turbo-model comments in config.py): only fires when the
# user's whole reply is essentially "I don't want to answer that", not when a
# real answer merely contains one of these words in passing.
_DECLINE_PHRASES = (
    "skip",
    "not sure",
    "dont know",
    "don't know",
    "no idea",
    "no clue",
    "you decide",
    "your call",
    "later",
    "pass",
    "n/a",
    "unsure",
    "havent decided",
    "haven't decided",
    "not decided",
    "whatever you think",
    "up to you",
)


def _is_decline(message: str) -> bool:
    normalized = message.strip().lower().strip(".!? ")
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _DECLINE_PHRASES)


def _trace(
    node_name: str,
    model_used: str | None,
    input_summary: str,
    output_summary: str,
    start: float,
    *,
    llm_input: list[dict] | None = None,
    llm_output: str | None = None,
) -> TraceEntry:
    return TraceEntry(
        node_name=node_name,
        model_used=model_used,
        input_summary=input_summary[:200],
        output_summary=output_summary[:200],
        duration_ms=int((time.perf_counter() - start) * 1000),
        llm_input=llm_input,
        llm_output=llm_output,
    )


_UNRESOLVED_CONNECTION_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT, TaskType.RETRIEVE_CONTEXT)
# Same confidence bar app.models.room_type_matches already uses for "is this
# the same room" (SequenceMatcher ratio 0.82) — rapidfuzz's 0-100 scale
# equivalent, kept consistent rather than picking a new number.
_ROOM_MATCH_THRESHOLD = 82


async def _existing_room_map(project_id: str) -> dict[str, str]:
    """room_id -> room-type name, for every LIVE room in the project. Shared
    by _resolve_room_hint below and _fallback_clarification_options — same
    query shape as app.context_builder._existing_rooms/app.retrieval.
    _existing_rooms, each of which already has its own copy rather than
    sharing one; kept consistent with that existing pattern."""
    from app import graph_store

    leaves = await graph_store.find_nodes(project_id, node_type="RoomType", lifecycle="active")
    return {leaf.room_id: str(leaf.value) for leaf in leaves if leaf.room_id and leaf.value is not None}


async def _resolve_room_hint(project_id: str, room_hint: Optional[str]) -> Optional[str]:
    """Fuzzy-matches an operation's raw room-name hint (app.tasks.TaskSpec.
    room_hint — sourced from app.llm.Operation.connection via
    canonical_mapper.split_connection) against existing LIVE rooms. A hint
    that matches no existing room (e.g. this is the first mention of that
    room) simply resolves to None — used only for the best-effort
    active_room_id guess below, never for write-target grounding (see
    app.pipeline, which mints a fresh room_id for a genuinely new room
    itself rather than fuzzy-matching first)."""
    if not room_hint:
        return None
    rooms = await _existing_room_map(project_id)
    best_room_id, best_score = None, 0
    for room_id, room_type in rooms.items():
        score = fuzz.ratio(room_hint.strip().lower(), room_type.lower())
        if score > best_score:
            best_room_id, best_score = room_id, score
    return best_room_id if best_score >= _ROOM_MATCH_THRESHOLD else None


async def _fallback_clarification_options(project_id: str) -> list[dict]:
    """Deterministic {"id", "label"} options used only when
    llm.resolve_operations' batched response is missing a "room" item for
    some task (a salvage recovered fewer items than were asked, or the
    response was simply short — see _generate_operation_questions below).
    The Resolution Agent prompt itself always returns at least one item per
    open operation on a clean response — this is purely a
    defensive backstop for a malformed/partial LLM response, filling the gap
    with the project's own live room list (same source as
    _existing_room_map) capped at 3 named rooms plus a catch-all."""
    rooms = await _existing_room_map(project_id)
    labels = list(dict.fromkeys(rooms.values()))[:3]
    options = [{"id": f"option_{i + 1}", "label": label} for i, label in enumerate(labels)]
    options.append({"id": f"option_{len(options) + 1}", "label": "Something else / a new room"})
    return options


async def _generate_operation_questions(project_id: str, tasks: list[TaskSpec]) -> tuple[list[dict], TraceEntry]:
    """ONE batched app.llm.resolve_operations call covering every task in
    `tasks` — fed the live project tree plus each task's own text/intent/
    connection/confusion. The Resolution Agent prompt (app.prompts.
    RESOLUTION_AGENT_SYSTEM_TEMPLATE) processes the whole operation list
    itself and returns, per operation `id`, a "room" ResolutionItem (when
    connection is None), a "content" ResolutionItem (when confusion is
    True), or both — correlated by id, not position.

    Only ONE open question is ever surfaced per operation per turn, even
    when both are open at once — content takes priority (deciding WHAT
    before WHERE). The other need simply re-triggers this same function
    again next turn once the surfaced one is answered (classify_intent_node
    re-runs the unresolved check after every merge, resume or fresh alike).
    This keeps the wire protocol at one op_id -> one open question, matching
    how the viewer submits an answer immediately per click rather than
    batching several answers into one reply — see PROMPT_CONTRACTS.md.

    Called only for write tasks classify_intent_node is holding the turn
    for (connection is None or confusion is True) — see _unresolved_tasks.
    A task the classifier couldn't ground, or left a content fork open on,
    must always be confirmed by the user; there is no model-confidence
    escape hatch."""
    tree_text = await context_builder.render_project_tree_text(project_id)
    operations = [
        {
            "id": task.op_id,
            "text": task.target,
            "intent": understanding.TASK_TYPE_TO_INTENT.get(task.type, task.type.value.lower()),
            "connection": task.connection,
            "confusion": task.confusion,
            "confusion_note": task.confusion_note,
        }
        for task in tasks
    ]

    capture: dict = {}
    call_start = time.perf_counter()
    results = await llm.resolve_operations(tree_text, operations, capture=capture)
    fallback_options = await _fallback_clarification_options(project_id)

    results_by_id: dict[str, list["llm.ResolutionItem"]] = {}
    for item in results:
        results_by_id.setdefault(item.id, []).append(item)

    questions: list[dict] = []
    resolved_count = 0
    for task in tasks:
        items = results_by_id.get(task.op_id, [])
        chosen = next((i for i in items if i.resolution_type == "content"), None) or (items[0] if items else None)

        if chosen is not None:
            resolved_count += 1
            question_text = chosen.question
            resolution_type = chosen.resolution_type
            if resolution_type == "room":
                # Precomputed here (app code), not by the model — see
                # RESOLVE_CONTEXT_CONFUSSION's own note that it copies a
                # room option's connection_path verbatim rather than
                # reconstructing it. A bundled multi-room option (room_ids
                # set) has no single connection_path of its own.
                options = [
                    {**o.model_dump(), "connection_path": (f"Rooms.{o.id}" if o.id and not o.room_ids else None)}
                    for o in chosen.options
                ]
            else:
                options = [{"label": o.label} for o in chosen.options]
        elif task.confusion:
            question_text = f'Regarding "{task.target}" — {task.confusion_note or "what would you like to do?"}'
            resolution_type = "content"
            options = [{"label": "Not sure yet"}]
        else:
            question_text = f'Which room is "{task.target}" for?'
            resolution_type = "room"
            options = fallback_options

        questions.append(
            {
                "op_id": task.op_id,
                "text": task.target,
                "question": question_text,
                "resolution_type": resolution_type,
                # Every room option is {"id", "label", "room_ids",
                # "connection_path"}, every content option is {"label"},
                # plus "allow_custom": true on the question itself — the
                # viewer must render both the option list AND a free-text
                # input for every clarifying question; app.chat's
                # operation_answers reply channel already accepts any
                # string for an op_id, whether it's an option's label or
                # something the user typed.
                "options": options,
                "allow_custom": True,
            }
        )

    output_summary = f"{len(questions)} operation(s) held for clarification"
    if resolved_count != len(tasks):
        output_summary += f" ({resolved_count}/{len(tasks)} resolved by the model, rest via fallback options)"
    entry = _trace(
        "resolve_operations",
        settings.model_intent_classifier,
        ", ".join(t.target for t in tasks),
        output_summary,
        call_start,
        llm_input=capture.get("messages"),
        llm_output=capture.get("raw_output"),
    )
    return questions, entry


def _unresolved_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """A write task still holds the turn while its room is ungrounded
    (connection is None) OR it carries a genuine unresolved content fork
    (confusion is True) — the two are independent (see app.llm.Operation.
    confusion's docstring) and either one alone is enough to block."""
    return [t for t in tasks if t.type in _UNRESOLVED_CONNECTION_TYPES and (t.connection is None or t.confusion)]


# Which task types can shift GraphState["active_room_id"] — a write clearly
# means the user is working on that room; a retrieval question about a room
# ("what's the budget for the kitchen?") counts too (it already requires a
# resolved connection like any other task per _UNRESOLVED_CONNECTION_TYPES).
# DIRECT_ANSWER/DATABASE_QUERY never carry a room connection and are
# deliberately excluded — plain chat/catalog lookups shouldn't hijack which
# room's questions get batched next.
_ACTIVE_ROOM_TASK_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT, TaskType.RETRIEVE_CONTEXT)


async def _resolve_active_room(project_id: str, tasks: list[TaskSpec]) -> Optional[str]:
    """Best-effort, pre-write guess at which room this turn is about —
    drives ONLY which room's open fields question_engine.find_knowledge_gaps
    batches next (see ChatSession.active_room_id's docstring), never
    write-target grounding. Walks `tasks` in their classified order and keeps
    the LAST one that resolves to a real existing room ("stick with the last
    provided one") — a task with no connection, or one naming a room that
    doesn't exist in the graph yet, is simply skipped rather than resetting
    the running result to None. Returns None when nothing in this turn
    resolves — the caller falls back to whatever was already active.

    The brand-new-room case (a task's connection names a room that doesn't
    exist until this turn's own write creates it) can't be resolved here —
    there's nothing to fuzzy-match yet. app.pipeline.run_pipeline corrects
    active_room_id from the room actually just written, but only as a
    fallback when this function came back empty."""
    resolved: Optional[str] = None
    for task in tasks:
        if task.type not in _ACTIVE_ROOM_TASK_TYPES:
            continue
        connection_room_id, connection_room_hint, _parent_instance = canonical_mapper.split_connection(task.connection)
        room_id = connection_room_id or await _resolve_room_hint(project_id, connection_room_hint or task.room_hint)
        if room_id:
            resolved = room_id
    return resolved


def _option_matches(answer: str, option: dict) -> bool:
    return answer == option.get("id") or answer == option.get("label")


def _matching_option(answer: str, options: list[dict]) -> Optional[dict]:
    return next((o for o in options if _option_matches(answer, o)), None)


async def _merge_resume_answers(
    project_id: str,
    tasks: list[TaskSpec],
    questions: list[dict],
    answers: dict[str, str],
    room_selections: dict[str, list[str]],
) -> tuple[list[TaskSpec], Optional[TraceEntry]]:
    """Resume-turn merge — see classify_intent_node's resume branch. A task
    with no stashed question, or no answer yet this turn, passes through
    unchanged (same op_id, same fields).

    A task whose open question was resolution_type "room" AND whose answer
    is a simple, exact match to one offered option (its id/label verbatim,
    or a bundled multi-room pick via room_selections — see app.llm.Option.
    room_ids) takes the existing cheap deterministic path: no LLM call, same
    behavior as before confusion existed. A bundled pick fans that ONE task
    into N clones, one per room id, each grounded to "Rooms.<room_id>" (the
    ids are already exact tree ids from the Resolution Agent, so no
    fuzzy room-hint matching needed the way a plain label answer goes
    through downstream). Clone op_ids get a "__<room_id>" suffix purely for
    trace/debugging readability — op_id is never read again once
    pending_operation_questions is cleared this turn.

    Every OTHER answered-but-open task — free text that doesn't exactly
    match an option, or ANY task whose open question was resolution_type
    "content" (a content decision always needs the model to fold the choice
    into clean operation text and clear `confusion`, even on an exact label
    match) — is batched into ONE app.llm.resolve_context_confusion call.
    Every operation that call returns is guaranteed connection non-null,
    confusion false; the merge below just copies those three fields onto
    the matching task by id."""
    questions_by_op_id = {q["op_id"]: q for q in questions}
    llm_items: list[dict] = []
    llm_task_ids: set[str] = set()

    merged: list[TaskSpec] = []
    for task in tasks:
        q = questions_by_op_id.get(task.op_id)
        answer = answers.get(task.op_id)
        if q is None or answer is None:
            merged.append(task)
            continue

        resolution_type = q.get("resolution_type", "room")
        room_ids = room_selections.get(task.op_id)
        matched_option = _matching_option(answer, q.get("options", [])) if resolution_type == "room" else None
        exact_match = resolution_type == "room" and (room_ids is not None or matched_option is not None)

        if exact_match:
            if room_ids:
                for room_id in room_ids:
                    clone = task.model_copy()
                    clone.op_id = f"{task.op_id}__{room_id}"
                    clone.connection = f"Rooms.{room_id}"
                    merged.append(clone)
            else:
                # Prefer the matched option's precomputed connection_path
                # (built from its real room id) over the raw answer text —
                # the viewer answers with an option's label, which is only
                # safe to use directly as a room hint when the option has no
                # backing room id (a new/inferred room, connection_path None).
                task.connection = (matched_option or {}).get("connection_path") or answer
                merged.append(task)
            continue

        llm_items.append(
            {
                "id": task.op_id,
                "original_operation": {
                    "text": task.target,
                    "intent": understanding.TASK_TYPE_TO_INTENT.get(task.type, task.type.value.lower()),
                    "connection": task.connection,
                    "confusion": task.confusion,
                    "confusion_note": task.confusion_note,
                },
                "pending_resolutions": [
                    {
                        "resolution_type": resolution_type,
                        "question": q.get("question", ""),
                        "options": q.get("options", []),
                        "user_answer": answer,
                    }
                ],
            }
        )
        llm_task_ids.add(task.op_id)
        merged.append(task)  # placeholder — overwritten below once resolved

    if not llm_items:
        return merged, None

    tree_text = await context_builder.render_project_tree_text(project_id)
    capture: dict = {}
    call_start = time.perf_counter()
    resolved = await llm.resolve_context_confusion(tree_text, llm_items, capture=capture)
    resolved_by_id = {r.id: r for r in resolved}

    final: list[TaskSpec] = []
    for task in merged:
        result = resolved_by_id.get(task.op_id) if task.op_id in llm_task_ids else None
        if result is not None:
            task.target = result.text
            task.connection = result.connection
            task.confusion = False
            task.confusion_note = None
        final.append(task)

    entry = _trace(
        "resolve_context_confusion",
        settings.model_intent_classifier,
        ", ".join(item["original_operation"]["text"] for item in llm_items),
        f"{len(resolved)}/{len(llm_items)} operation(s) resolved",
        call_start,
        llm_input=capture.get("messages"),
        llm_output=capture.get("raw_output"),
    )
    return final, entry


async def classify_intent_node(state: GraphState) -> dict:
    """Thin wrapper around app.understanding.understand() — see
    ARCHITECTURE_BASELINE.md Phase 1. Understanding itself (classification +
    guards) lives in that pure, DB-free module; this node's only jobs are
    threading the LLM capture through into the trace and storing `tasks`,
    which routing reads.

    On a resume (pending_operation_questions from a prior turn),
    classify_operations is skipped entirely (no LLM call unless a free-text
    or content answer needs one), the stashed tasks are reloaded, and
    `operation_answers` (ChatRequest's structured {op_id: chosen_value}
    reply) is merged into their `connection`/`confusion` fields via
    _merge_resume_answers. An op_id answered with a bundled multi-room
    option (operation_room_selections[op_id] set) fans that ONE task out
    into one task per room id instead of setting a single connection — see
    _merge_resume_answers' docstring. Either way (fresh classification or
    resume), if any write task still has connection=None or confusion=True
    afterward, this returns pending_operation_questions instead of
    proceeding — a partially-answered resume re-asks only the
    still-unresolved ops, everything already answered stays answered.

    Also owns decline-detection: if this message declines the field
    `current_field` says was just asked about, and that field was
    room-scoped, the room is added to `skipped_rooms` right here — a plain
    state mutation, not a routing branch. It does NOT short-circuit the
    turn: classification/routing below proceeds exactly as it would have, so
    the turn's actual operation still runs. app.pipeline.run_pipeline picks
    up the updated skipped_rooms afterward and naturally asks about
    something else."""
    start = time.perf_counter()
    with _node_span("classify_intent") as span:
        current_field = state.get("current_field") or {}
        skipped_rooms = state.get("skipped_rooms") or []
        declined_room_id = current_field.get("room_id")
        if declined_room_id and _is_decline(state["message"]) and declined_room_id not in skipped_rooms:
            skipped_rooms = [*skipped_rooms, declined_room_id]

        pending = state.get("pending_operation_questions")
        capture: dict = {}
        extra_trace: list[TraceEntry] = []
        if pending:
            answers = state.get("operation_answers") or {}
            room_selections = state.get("operation_room_selections") or {}
            tasks = [TaskSpec(**t) for t in pending["tasks"]]
            tasks, resolve_entry = await _merge_resume_answers(
                state["project_id"], tasks, pending.get("questions") or [], answers, room_selections
            )
            if resolve_entry:
                extra_trace.append(resolve_entry)
            raw_intents = [understanding.TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
            output_summary = f"resumed {len(tasks)} operation(s), {len(answers)} answer(s) applied"
        else:
            tree_text = await context_builder.render_project_tree_text(state["project_id"])
            meaning = await understanding.understand(
                state["message"],
                state["history"],
                pending_field=current_field.get("canonical_paths"),
                tree_text=tree_text,
                capture=capture,
            )
            tasks = meaning.tasks
            raw_intents = [understanding.TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
            output_summary = ",".join(raw_intents)
            if raw_intents != meaning.raw_intents:
                output_summary += f" (guard adjusted from: {','.join(meaning.raw_intents)})"

        entry = _trace(
            "classify_intent",
            settings.model_intent_classifier,
            state["message"],
            output_summary,
            start,
            llm_input=capture.get("messages"),
            llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)

    # Checked on EVERY turn — no graph write is ever allowed to proceed with
    # connection=None (e.g. a bare "add a laminate", no room in scope:
    # classify_operations can't ground it, and the turn must wait for the
    # user rather than guess). No model-confidence escape hatch: whenever
    # this block runs at all, the turn is held — see
    # _generate_operation_questions, which guarantees a question for every
    # task passed to it.
    unresolved = _unresolved_tasks(tasks)
    if unresolved:
        with _node_span("generate_operation_questions") as cspan:
            questions, clarify_entry = await _generate_operation_questions(state["project_id"], unresolved)
            cspan.update(
                input=", ".join(t.target for t in unresolved),
                output=f"{len(questions)} operation(s) held for clarification",
            )
        return {
            "intent": raw_intents,
            "tasks": tasks,
            "pending_operation_questions": {"tasks": [t.model_dump() for t in tasks], "questions": questions},
            "skipped_rooms": skipped_rooms,
            "trace": [entry, *extra_trace, clarify_entry],
        }
    # Only computed once the turn actually proceeds (never on the
    # still-unresolved branch above — nothing's grounded yet to derive a
    # room from).
    active_room_id = await _resolve_active_room(state["project_id"], tasks) or state.get("active_room_id")
    return {
        "intent": raw_intents,
        "tasks": tasks,
        "pending_operation_questions": None,
        "skipped_rooms": skipped_rooms,
        "active_room_id": active_room_id,
        "trace": [entry, *extra_trace],
    }


async def run_pipeline_node(state: GraphState) -> dict:
    """Thin wrapper (span + trace) around app.pipeline.run_pipeline — the
    single orchestrator that replaces every per-intent node this graph used
    to route to (build_context/delete_context/retrieve_context/query_catalog/
    generate_answer/generate_question/validate_completeness/complete_project/
    handle_split_intents). Runs for every turn that reaches it, whether the
    turn classified into one task or several — see app.pipeline's module
    docstring for the dependency/ordering rules between its internal steps."""
    start = time.perf_counter()
    with _node_span("run_pipeline") as span:
        result = await pipeline.run_pipeline(state)
        entry = _trace(
            "run_pipeline",
            None,
            state["message"],
            result.get("next_message") or result.get("answer") or "",
            start,
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    result["trace"] = [entry, *result.get("trace", [])]
    return result


async def describe_image_node(state: GraphState, image_url: str) -> tuple[TraceEntry, str]:
    start = time.perf_counter()
    with _node_span("describe_image") as span:
        capture: dict = {}
        description = await llm.vision(image_url, state["message"] or prompts.DEFAULT_VISION_PROMPT, capture=capture)
        entry = _trace(
            "describe_image", settings.model_vision, image_url, description, start,
            llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    return entry, description


def _route_intent(state: GraphState) -> str:
    """classify_intent_node already short-circuited this turn (one or more
    write tasks still have connection=None) whenever pending_operation_questions
    is set — nothing executes this turn; the graph ends immediately so the
    client sees the clarification question. Otherwise every turn — single
    task or several — runs through the one pipeline node."""
    if state.get("pending_operation_questions"):
        return "operation_questions_pending"
    return "pipeline"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("run_pipeline", run_pipeline_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {"pipeline": "run_pipeline", "operation_questions_pending": END},
    )
    graph.add_edge("run_pipeline", END)

    return graph.compile()


app_graph = build_graph()
