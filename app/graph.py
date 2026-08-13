import operator
import re
import time
from contextlib import contextmanager
from typing import Annotated, Optional, TypedDict

from langfuse import get_client
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from rapidfuzz import fuzz

from app import calculation, canonical_mapper, context_builder, llm, execution, graph_store, inference, prompts, rag, retrieval, understanding, versioning
from app.config import settings
from app.context_builder import build_context
from app.models import FIELD_LABELS, FIELD_TIERS, ProjectContext, TraceEntry
from app.question_engine import KnowledgeGapBatch, find_knowledge_gaps, generate_question
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
    # decline-detection step and question_engine.find_knowledge_gaps'
    # auto-advance. A room only ever moves INTO this list; there's no
    # explicit "unskip" (see find_knowledge_gaps' docstring note).
    skipped_rooms: list[str]
    # {"canonical_paths", "room_id"} for the field(s) the LAST question
    # batch was about — written only by generate_question_node, read only by
    # classify_intent_node (feeds classify_operations' pending-field context
    # and is what decline-detection keys off, via room_id). Deliberately NOT
    # pending_gap — see that field's own comment below for why the two are
    # kept separate.
    current_field: Optional[dict]
    # Which room a batched question/write is currently focused on — see
    # ChatSession.active_room_id's docstring (app/models.py) for the full
    # contract. Written by classify_intent_node (from the turn's resolved
    # tasks), corrected by build_context_node for a brand-new room its own
    # write just created, and surfaced forward by generate_question_node
    # whenever question_engine.find_knowledge_gaps auto-advances it. Never
    # used for write-target grounding — that stays exclusively
    # task.connection/room_hint, unchanged.
    active_room_id: Optional[str]
    intent: list[str]
    tasks: list[TaskSpec]
    retrieved: str
    pending_question: Optional[str]
    # Serialized question_engine.KnowledgeGapBatch for the currently open
    # gap(s). Written by build_context_node right after a value commits, and
    # by generate_question_node right after it computes the batch it's about
    # to ask about — no other node reads or writes this. See PENDING_GAP_ANALYSIS.md.
    pending_gap: Optional[dict]
    # Set by classify_intent_node when classify_operations returned a
    # write task (EDIT_CONTEXT/DELETE_CONTEXT) with connection=None — one
    # clarifying multiple-choice question per unresolved op, batched
    # together (see the classifier-connection plan). Shaped
    # {"tasks": [TaskSpec.model_dump(), ...], "questions": [{"op_id", "text",
    # "question", "options"}, ...]} — stashes the WHOLE task list (resolved
    # and unresolved alike) so the resume turn never needs to re-classify.
    pending_operation_questions: Optional[dict]
    # {op_id: chosen_value} from the client, answering a prior turn's
    # pending_operation_questions — read by classify_intent_node's resume
    # branch, never set by any graph node itself.
    operation_answers: Optional[dict[str, str]]
    update_summary: Optional[str]
    answer: str
    complete: bool
    # Set by handle_split_intents_node when a multi-intent turn also had an
    # info-ask alongside an EDIT_CONTEXT task — see _route_post_completeness.
    needs_answer: bool
    # True once some node has already produced this turn's trailing
    # ask-question/confirmation/wrap-up — analyze_context_node (which runs on
    # every branch) checks this to avoid a second, redundant generation call
    # for the same turn.
    question_generated: bool
    wrapup_message: Optional[str]
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


# Deliberately narrow and deterministic (no extra LLM call), same posture as
# _is_decline above — a reply to a yes/no confirmation is short and direct
# far more often than not. An ambiguous reply (neither list matches) is

# — never silently apply an unconfirmed change.
_AFFIRMATIVE_PHRASES = ("yes", "yeah", "yep", "yup", "correct", "that's right", "confirm", "go ahead", "affirmative")
_NEGATIVE_PHRASES = (
    "no", "nope", "don't", "dont", "keep the old", "keep it", "leave it", "never mind", "cancel", "wrong",
)


def _is_affirmative(message: str) -> bool:
    normalized = message.strip().lower().strip(".!? ")
    return any(phrase in normalized for phrase in _AFFIRMATIVE_PHRASES) and not _is_negative(message)


def _is_negative(message: str) -> bool:
    normalized = message.strip().lower().strip(".!? ")
    return any(phrase in normalized for phrase in _NEGATIVE_PHRASES)


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


_UNRESOLVED_CONNECTION_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT,TaskType.RETRIEVE_CONTEXT)


async def _fallback_clarification_options(project_id: str) -> list[dict]:
    """Deterministic {"id", "label"} options used only when
    llm.resolve_room_connections' batched response is missing an entry for
    some task (a salvage recovered fewer items than were asked, or the
    response was simply short — see _generate_operation_questions below).
    The Room Resolution Agent prompt itself always returns a question + at
    least one option per operation on a clean response (no
    "resolvable without asking" verdict to fall back from anymore, unlike
    the old generate_clarification_question) — this is purely a defensive
    backstop for a malformed/partial LLM response, filling the gap with the
    project's own live room list (same source as _existing_room_map) capped
    at 3 named rooms plus a catch-all, mirroring the shape of a real
    model-authored option list."""
    rooms = await _existing_room_map(project_id)
    labels = list(dict.fromkeys(rooms.values()))[:3]
    options = [{"id": f"option_{i + 1}", "label": label} for i, label in enumerate(labels)]
    options.append({"id": f"option_{len(options) + 1}", "label": "Something else / a new room"})
    return options


async def _generate_operation_questions(project_id: str, tasks: list[TaskSpec]) -> tuple[list[dict], TraceEntry]:
    """ONE batched app.llm.resolve_room_connections call covering every task
    in `tasks` — fed the live project tree plus each task's own text/intent
    — replacing the old per-task asyncio.gather of individual
    generate_clarification_question calls. The Room Resolution Agent prompt
    (app.prompts.ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE) processes the whole
    operation list itself and always returns a question + at least one
    option per operation, in the same order — so every task in `tasks` is
    guaranteed a question in the returned list with no per-task confidence
    branching needed; _fallback_clarification_options only covers the
    defensive case where the response comes back short. Called only for
    write tasks whose connection is still None after classification/merge —
    see classify_intent_node, which unconditionally holds the WHOLE turn's
    writes back whenever this is called at all: a task the classifier
    couldn't ground on its own (connection=None) must always be confirmed by
    the user, not silently resolved via active_room_id/room_hint guesswork
    (see the classifier-connection plan and its "always block on touch"
    follow-up).

    Returns ONE TraceEntry for the whole batched call (llm_input/llm_output
    via resolve_room_connections' `capture` param) rather than one per
    task — there's only one real LLM call now, so there's only one prompt/
    response for the viewer's Process Trace panel to show."""
    tree_text = await context_builder.render_project_tree_text(project_id)
    operations = [
        {
            "text": task.target,
            "intent": understanding.TASK_TYPE_TO_INTENT.get(task.type, task.type.value.lower()),
            "connection": None,
        }
        for task in tasks
    ]

    capture: dict = {}
    call_start = time.perf_counter()
    results = await llm.resolve_room_connections(tree_text, operations, capture=capture)
    fallback_options = await _fallback_clarification_options(project_id)

    questions: list[dict] = []
    for i, task in enumerate(tasks):
        result = results[i] if i < len(results) else None
        if result is not None and result.question and result.options:
            question_text = result.question
            options = [o.model_dump() for o in result.options]
        else:
            question_text = f'Which room is "{task.target}" for?'
            options = fallback_options
        questions.append(
            {
                "op_id": task.op_id,
                "text": task.target,
                "question": question_text,
                # Every option is {"id", "label"} (see
                # app.llm.RoomResolutionOption) plus "allow_custom": true on
                # the question itself — the viewer must render both the
                # option list AND a free-text input for every clarifying
                # question; app.chat's operation_answers reply channel
                # already accepts any string for an op_id, whether it's an
                # option's label or something the user typed.
                "options": options,
                "allow_custom": True,
            }
        )

    output_summary = f"{len(questions)} operation(s) held for clarification"
    if len(results) != len(tasks):
        output_summary += f" ({len(results)}/{len(tasks)} resolved by the model, rest via fallback options)"
    entry = _trace(
        "resolve_room_connections",
        settings.model_intent_classifier,
        ", ".join(t.target for t in tasks),
        output_summary,
        call_start,
        llm_input=capture.get("messages"),
        llm_output=capture.get("raw_output"),
    )
    return questions, entry


def _unresolved_connection_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    return [t for t in tasks if t.type in _UNRESOLVED_CONNECTION_TYPES and t.connection is None]


async def classify_intent_node(state: GraphState) -> dict:
    """Thin wrapper around app.understanding.understand() — see
    ARCHITECTURE_BASELINE.md Phase 1. Understanding itself (classification +
    guards) lives in that pure, DB-free module; this node's only jobs are
    threading the LLM capture through into the trace and storing `tasks`,
    which routing reads (Phase 2 cutover).

    — classify_operations is skipped entirely (no LLM call), the stashed
    tasks are reloaded, and `operation_answers` (ChatRequest's structured
    {op_id: chosen_value} reply) is merged into their `connection` fields.
    Either way (fresh classification or resume), if any write task still has
    connection=None afterward, this returns pending_operation_questions
    instead of proceeding to normal routing (_route_intent short-circuits on
    it) — a partially-answered resume re-asks only the still-unresolved
    ops, everything already answered stays answered.

    Also owns decline-detection (see PENDING_GAP_ANALYSIS.md and the
    pending_gap-centralization plan): if this message declines the field
    `current_field` says was just asked about, and that field was
    room-scoped, the room is added to `skipped_rooms` right here — a plain
    state mutation, not a routing branch. It does NOT short-circuit the
    turn: classification/routing below proceeds exactly as it would have,
    so the turn's actual operation (DIRECT_ANSWER/CONTEXT_UPDATE/etc.)
    still runs. generate_question_node picks up the updated skipped_rooms
    afterward and naturally asks about something else."""
    start = time.perf_counter()
    with _node_span("classify_intent") as span:
        current_field = state.get("current_field") or {}
        skipped_rooms = state.get("skipped_rooms") or []
        declined_room_id = current_field.get("room_id")
        if declined_room_id and _is_decline(state["message"]) and declined_room_id not in skipped_rooms:
            skipped_rooms = [*skipped_rooms, declined_room_id]

        pending = state.get("pending_operation_questions")
        capture: dict = {}
        if pending:
            answers = state.get("operation_answers") or {}
            tasks = [TaskSpec(**t) for t in pending["tasks"]]
            for task in tasks:
                if task.op_id in answers:
                    task.connection = answers[task.op_id]
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
            # TASK_TYPE_TO_INTENT is a full bijection over every TaskType now
            # (classify_operations' CONTEXT_DELETE is a real classifier label,
            # unlike the old guard_delete override) — this list is telemetry
            # only (_route_intent routes on state["tasks"] directly, never
            # this); the .get(..., fallback) is just defensive.
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

    # Checked on EVERY turn now, single-op included — no graph write is ever
    # allowed to proceed with connection=None (e.g. a bare "add a laminate",
    # single operation, no room in scope: classify_operations can't ground
    # it, and the turn must wait for the user rather than guess). This used
    # to be gated to multi-op turns only (a single operation's own
    # extraction was trusted to resolve its own room) — that gate is gone.
    # Unconditional, with no model-confidence escape hatch: whenever this
    # block runs at all, the turn is held — see _generate_operation_questions,
    # which now guarantees a question for every task passed to it, so
    # `questions` below is never empty when `unresolved` isn't. A task the
    # classifier itself couldn't ground (connection=None) always gets
    # confirmed by the user before any write proceeds, whether it's alone or
    # part of a batch.
    unresolved = _unresolved_connection_tasks(tasks)
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
            "question_generated": True,
            "trace": [entry, clarify_entry],
        }
    # Only computed once the turn actually proceeds (never on the
    # still-unresolved branch above — nothing's grounded yet to derive a
    # room from). See _resolve_active_room's own docstring for the "stick
    # with the last provided one" rule and why a brand-new room's own
    # not-yet-existing-so-not-fuzzy-matchable case is left to
    # build_context_node/delete_context_node's post-write fallback instead.
    active_room_id = await _resolve_active_room(state["project_id"], tasks) or state.get("active_room_id")
    return {
        "intent": raw_intents,
        "tasks": tasks,
        "pending_operation_questions": None,
        "skipped_rooms": skipped_rooms,
        "active_room_id": active_room_id,
        "trace": [entry],
    }


async def retrieve_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    # Uses this task's own operation text when the turn was split into
    # several operations by classify_operations (e.g. a CONTEXT_RETRIEVAL and
    # a DATABASE_RETRIEVAL in the same message) — falls back to the whole
    # message for the common single-operation case, same as
    # build_context_node/delete_context_node.
    message = (task.target if task else None) or state["message"]
    start = time.perf_counter()
    with _node_span("retrieve_context") as span:
        nodes = await retrieval.retrieve_scoped(message, state["project_id"])
        parts = []
        for node in nodes:
            if node.value is None:
                continue
            field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type, node.node_type)
            parts.append(f"{field_name}={node.value}")
        summary = ", ".join(parts) or "no context yet"
        entry = _trace("retrieve_context", None, message, summary, start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"retrieved": summary, "trace": [entry]}


async def query_catalog_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    # See retrieve_context_node's comment above — same per-operation target
    # fallback, so a DATABASE_RETRIEVAL operation searches on its own text
    # rather than the whole turn's message when the turn has more than one
    # operation.
    message = (task.target if task else None) or state["message"]
    start = time.perf_counter()
    with _node_span("query_catalog") as span:
        items = await rag.query_catalog(message)
        summary = "; ".join(f"{i.title}: {i.description[:80]}" for i in items) or "no matches"
        entry = _trace("query_catalog", settings.model_embedding, message, summary, start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"retrieved": summary, "trace": [entry]}


async def _summarize_written(project_id: str, written_paths: list[str]) -> str:
    """Deterministic "Got it — noted ..." confirmation of what build_context
    just wrote — no LLM call, so it doesn't add to per-turn latency. Materials
    leaves and freeform/unmapped paths are summarized generically rather than
    itemized (unlike the old PartialContext-era version) — see
    app.context_builder.LEAF_TO_FIELD_NAME, which only covers structured
    slot fields."""
    if not written_paths:
        return ""
    nodes = await graph_store.find_nodes(project_id)
    by_path = {n.canonical_path: n for n in nodes}

    parts: list[str] = []
    materials_noted = False
    other_count = 0
    for path in written_paths:
        node = by_path.get(path)
        if node is None or node.value is None:
            continue
        if ".Materials." in path:
            materials_noted = True
            continue
        field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type)
        if field_name is None:
            other_count += 1
            continue
        parts.append(f"{FIELD_LABELS.get(field_name, field_name)}: {node.value}")

    if materials_noted:
        parts.append("materials noted")
    if other_count:
        parts.append(f"{other_count} other detail{'s' if other_count != 1 else ''} noted")
    if not parts:
        return ""
    return "Got it — noted " + ", ".join(parts) + "."


async def build_context_node(
    state: GraphState, task: Optional[TaskSpec] = None, resolved: Optional[context_builder.ResolvedBuild] = None
) -> dict:
    """
    `task` is explicitly passed by app.execution.execute() (a multi-operation
    turn's per-clause task — see app.understanding.understand); the
    single-task path routed directly from classify_intent calls this with no
    second argument, so it's recovered here from state["tasks"][0] instead —
    _route_intent only ever routes straight to this node when len(tasks) == 1,
    so that's always the SAME task classify_intent_node/_generate_operation_questions
    already resolved a connection for. Room grounding comes ONLY from
    `task.connection`/`task.room_hint` now — there is no more active_room_id
    fallback (removed system-wide; see PENDING_GAP_ANALYSIS.md). A turn with
    no explicit room mention and no resolvable connection falls through to
    context_builder.resolve_context's own "create an unlabeled room" fallback
    instead of continuing whatever room the previous turn touched.

    `resolved` (app.context_builder.ResolvedBuild) lets a caller that already
    ran resolve_context() — app.execution.execute()'s clustering, for the
    FIRST task committed in a write-cluster — skip straight to
    commit_context() instead of resolving again. Every other caller (the
    single-task path, and every task after the first within a cluster) leaves
    this None and gets the full resolve_context()+commit_context() build_context()
    already did, which is what keeps a later same-cluster commit correct
    against whatever an earlier one in the same cluster just wrote (see
    app.execution._commit_cluster). `task.connection` (see
    canonical_mapper.split_connection) takes priority over `task.room_hint`
    here too, exactly like app.execution._resolve_write_task's own parallel
    resolve pass — required so a task re-resolving fresh (every task after
    the first in a cluster) doesn't lose its connection grounding just
    because it isn't reusing a pre-computed `resolved` plan.

    On a successful commit (no held-back conflict), this is also the ONE
    place — besides generate_question_node itself — that writes pending_gap:
    it recomputes find_knowledge_gaps right after the write lands (scoped to
    whichever room this write actually touched — see active_room_id below),
    so the cache reflects the fact that was just committed rather than
    waiting for generate_question_node to notice on its own next run.

    Also corrects `active_room_id` (see ChatSession.active_room_id's
    docstring) for the one case classify_intent_node's own pre-write
    `_resolve_active_room` can't cover: a brand-new room, created by THIS
    write, that didn't exist yet to fuzzy-match against. `connection_room_id`
    (a grounded, already-existing room) takes priority when present — same
    room classify_intent_node itself would have resolved, kept here only for
    consistency; otherwise `result.room_id` (the room this write actually
    just committed to, new or not) is authoritative and overrides whatever
    stale room_id carried over in state, since a write with no grounded
    existing-room connection is exactly the "this turn is now about a
    different room" case. Only if this write touched no room at all
    (a pure Timeline/Budget/freeform Project-level fact) does the old
    state value survive untouched."""
    start = time.perf_counter()
    task = task or next(iter(state.get("tasks") or []), None)
    message = (task.target if task else None) or state["message"]
    print("build_context_node: state=", state.get("tasks"))
    print("build_context_node: message=", message, "task=", task, "resolved=", resolved)
    connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection if task else None)
    room_hint = connection_room_hint or (task.room_hint if task else None)
    with _node_span("build_context") as span:
        if resolved is not None:
            result = await context_builder.commit_context(resolved)
        else:
            result = await build_context(message, state["project_id"], connection_room_id, room_hint=room_hint)
        entry = _trace(
            "build_context",
            settings.model_extraction,
            message,
            f"wrote {len(result.written)} node(s), ",
            start,
        )
        span.update(input=entry.input_summary, output=entry.output_summary)

    active_room_id = connection_room_id or result.room_id or state.get("active_room_id")
    update_summary = await _summarize_written(state["project_id"], result.written)
    gap_batch = await find_knowledge_gaps(state["project_id"], active_room_id, state.get("skipped_rooms") or [])
    return {
        "update_summary": update_summary or None,
        "pending_gap": gap_batch.model_dump() if gap_batch else None,
        "active_room_id": active_room_id,
        "trace": [entry],
    }



# Delete's core-noun-phrase extraction — strips the trigger verb and common
# trailing boilerplate so what's left is comparable, whole-string, against a
# room name (see _match_room_for_deletion). Deliberately whole-string
# fuzz.ratio there, not fuzz.partial_ratio (retrieval.py's tool for "is this
# room mentioned anywhere in this text") — partial_ratio would score "remove
# the ceiling fan from the living room" just as high as "remove the living
# room" purely because "living room" appears as a substring in both, which
# would misfire a room-cascade deletion for what's actually an item-level one.
_DELETE_CORE_PHRASE_RE = re.compile(
    r"\b(?:remove|delete|cancel|get rid of|take out|drop|scratch)\b\s*(?:the\s+)?(.+?)"
    r"(?:\s+from\s+(?:the\s+)?project|\s+from\s+the\s+list|\s+altogether)?\s*[.!?]*$",
    re.IGNORECASE,
)
# Same confidence bar app.models.room_type_matches already uses for "is this
# the same room" (SequenceMatcher ratio 0.82) — rapidfuzz's 0-100 scale
# equivalent, kept consistent rather than picking a new number.
_ROOM_DELETE_MATCH_THRESHOLD = 82


async def _existing_room_map(project_id: str) -> dict[str, str]:
    """room_id -> room-type name, for every LIVE room in the project. Shared
    by _match_room_for_deletion (is target_text itself naming a room?) and
    _resolve_room_hint (does a segmented clause's room_hint name an existing
    room?) below — same query shape as context_builder._existing_rooms/
    retrieval._existing_rooms, each of which already has its own copy rather
    than sharing one; kept consistent with that existing pattern instead of
    reaching into another module's `_`-prefixed helper."""
    leaves = await graph_store.find_nodes(project_id, node_type="RoomType", lifecycle="active")
    return {leaf.room_id: str(leaf.value) for leaf in leaves if leaf.room_id and leaf.value is not None}


async def _match_room_for_deletion(project_id: str, target_text: str) -> Optional[str]:
    """None unless target_text's core phrase (after _DELETE_CORE_PHRASE_RE
    strips the trigger word and trailing boilerplate) is itself close, as a
    WHOLE string, to an existing room's name — see the module comment above
    for why this guards against item-level deletions inside a named room
    being misread as deleting the room itself."""
    match = _DELETE_CORE_PHRASE_RE.search(target_text)
    core = (match.group(1) if match else target_text).strip().strip(".!?")
    if core.lower().startswith("the "):
        core = core[4:].strip()
    if not core:
        return None
    rooms = await _existing_room_map(project_id)
    best_room_id, best_score = None, 0
    for room_id, room_type in rooms.items():
        score = fuzz.ratio(core.lower(), room_type.lower())
        if score > best_score:
            best_room_id, best_score = room_id, score
    return best_room_id if best_score >= _ROOM_DELETE_MATCH_THRESHOLD else None


async def _resolve_room_hint(project_id: str, room_hint: Optional[str]) -> Optional[str]:
    """Fuzzy-matches an operation's raw room-name hint (app.tasks.TaskSpec.
    room_hint — sourced from app.llm.Operation.connection via
    canonical_mapper.split_connection, not auto-detected) against existing
    LIVE rooms, same confidence bar as _match_room_for_deletion's own
    room-identity check. A hint that matches no existing room (e.g. this is
    the first mention of that room) simply resolves to None."""
    if not room_hint:
        return None
    rooms = await _existing_room_map(project_id)
    best_room_id, best_score = None, 0
    for room_id, room_type in rooms.items():
        score = fuzz.ratio(room_hint.strip().lower(), room_type.lower())
        if score > best_score:
            best_room_id, best_score = room_id, score
    return best_room_id if best_score >= _ROOM_DELETE_MATCH_THRESHOLD else None


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
    provided one," confirmed with the user) — a task with no connection, or
    one naming a room that doesn't exist in the graph yet (nothing to
    fuzzy-match against something not there), is simply skipped rather than
    resetting the running result to None. Returns None (not the old
    active_room_id) when nothing in this turn resolves — callers fall back
    to whatever was already active themselves, same pattern
    classify_intent_node's own caller uses.

    The brand-new-room case (a task's connection names a room that doesn't
    exist until this turn's own write creates it) can't be resolved here —
    there's nothing to fuzzy-match yet. build_context_node/delete_context_node
    correct active_room_id from the room actually just written, but only as a
    fallback when this function came back empty, so that post-write
    correction never overrides an already-determined "last mentioned"
    result."""
    resolved: Optional[str] = None
    for task in tasks:
        if task.type not in _ACTIVE_ROOM_TASK_TYPES:
            continue
        connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection)
        room_id = connection_room_id or await _resolve_room_hint(project_id, connection_room_hint or task.room_hint)
        if room_id:
            resolved = room_id
    return resolved


class DeleteTargetPreview(BaseModel):
    """Read-only preview of what a DELETE_CONTEXT task would touch — see
    resolve_delete_target. canonical_path is the specific node (item- or
    room-level) a confident match resolved to, if any. room_id is that
    target's OWN room scope, exposed separately from canonical_path so a
    caller building connectivity keys (app.execution._resolve_write_task)
    can cluster this delete with an EDIT_CONTEXT task that shares the same
    room but a different leaf path (e.g. "remove the sofa" — a
    Furniture.sofa path — vs a separate "set the kitchen style" write,
    a RoomType/Style path — both scoped to the same room, sharing no leaf
    path in common)."""

    canonical_path: Optional[str] = None
    room_id: Optional[str] = None


async def resolve_delete_target(state: GraphState, task: Optional[TaskSpec] = None) -> DeleteTargetPreview:
    """Read-only preview of what a DELETE_CONTEXT task would touch — used by
    app.execution.execute()'s clustering to detect when a delete and another
    operation in the same turn are "connected" before deciding write order.
    Reuses the exact same pure lookups delete_context_node itself uses
    (_match_room_for_deletion, canonical_mapper.resolve_deletion_target); a
    room-level match previews the room's own path; it's the right
    connectivity key even for a preview taken before delete_context_node's
    own real retraction runs, since the cascade touches everything under it.
    "no_match"/"ambiguous" preview
    nothing: neither touches anything this turn, so there's nothing to be
    connected to. `task.connection` (an anchor hint, not a final write path —
    see canonical_mapper.split_connection) takes priority over the older
    regex-detected task.room_hint: a grounded connection's room segment
    overrides room_id outright, an ungrounded (free-text) connection
    overrides the room_hint that's fuzzy-matched below."""
    target_text = (task.target if task else None) or state["message"]
    connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection if task else None)
    room_id = connection_room_id or await _resolve_room_hint(
        state["project_id"], connection_room_hint or (task.room_hint if task else None)
    )

    room_match = await _match_room_for_deletion(state["project_id"], target_text)
    if room_match is not None:
        return DeleteTargetPreview(canonical_path=f"Project.Rooms.{room_match}", room_id=room_match)

    match = await canonical_mapper.resolve_deletion_target(target_text, state["project_id"], room_id)
    if match.outcome == "single":
        instance = match.match.instance
        return DeleteTargetPreview(canonical_path=instance.canonical_path, room_id=instance.room_id)
    return DeleteTargetPreview()


async def delete_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    """The write path for a DELETE_CONTEXT task (see app.understanding.guard_delete).
    Checks for a room-level deletion first (see _match_room_for_deletion) and
    retracts the whole room's subtree immediately; otherwise falls through to
    item-level matching among live freeform facts
    (canonical_mapper.resolve_deletion_target) and retracts that node
    immediately — never fabricates a match: no confident match asks for
    clarification, an ambiguous one lists the candidates instead of guessing.
    No confirmation gate on either path (see PENDING_GAP_ANALYSIS.md) — a
    matched deletion always applies. `task.room_hint` is a raw room-type NAME
    (app.tasks.TaskSpec), not a room_id — resolved here via
    _resolve_room_hint's fuzzy match against existing rooms (no more
    active_room_id fallback — removed system-wide). `task.connection` takes
    priority over `task.room_hint` the same way resolve_delete_target's own
    docstring describes. `task` defaults to state["tasks"][0] when the graph
    routes here directly for a single-task turn (task=None) — see
    build_context_node's matching comment."""
    start = time.perf_counter()
    task = task or next(iter(state.get("tasks") or []), None)
    target_text = (task.target if task else None) or state["message"]
    connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection if task else None)
    room_id = connection_room_id or await _resolve_room_hint(
        state["project_id"], connection_room_hint or (task.room_hint if task else None)
    )

    with _node_span("delete_context") as span:
        room_match = await _match_room_for_deletion(state["project_id"], target_text)
        if room_match is not None:
            room_node = await graph_store.find_one(state["project_id"], f"Project.Rooms.{room_match}")
            if room_node is None:
                entry = _trace("delete_context", None, room_match, "room match had no container node — skipped", start)
                span.update(input=entry.input_summary, output=entry.output_summary)
                return {"trace": [entry]}
            room_type_node = await graph_store.find_one(state["project_id"], f"Project.Rooms.{room_match}.RoomType")
            room_label = str(room_type_node.value) if room_type_node and room_type_node.value else f"room {room_match}"
            retracted = await versioning.retract_subtree(room_node)
            entry = _trace(
                "delete_context", None, target_text,
                f"retracted room {room_label!r} and {len(retracted) - 1} descendant node(s)", start,
            )
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"update_summary": f"Removed {room_label!r}.", "trace": [entry]}

        match = await canonical_mapper.resolve_deletion_target(target_text, state["project_id"], room_id)

        if match.outcome == "none":
            question = f'I couldn\'t find "{target_text}" to remove — can you tell me exactly what it\'s called?'
            entry = _trace("delete_context", None, target_text, "no confident match — asked for clarification", start)
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"pending_question": question, "question_generated": True, "trace": [entry]}

        if match.outcome == "ambiguous":
            labels = ", ".join(c.label for c in match.candidates)
            question = f'I found more than one match for "{target_text}" — did you mean: {labels}?'
            entry = _trace(
                "delete_context", None, target_text, f"ambiguous match ({len(match.candidates)} candidates) — asked for clarification", start
            )
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"pending_question": question, "question_generated": True, "trace": [entry]}

        node = match.match.instance
        label = match.match.label
        await versioning.retract_node(node)
        entry = _trace("delete_context", None, target_text, f"retracted {node.canonical_path}", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
        return {"update_summary": f"Removed {label!r}.", "trace": [entry]}


async def handle_split_intents_node(state: GraphState) -> dict:
    """Only reached when classify_intent found more than one distinct ask in
    the message (see _route_intent). Delegates to app.execution.execute(),
    which resolves/clusters/commits write tasks and dispatches read tasks
    concurrently — not LangGraph fan-out — then merges results back into a
    single partial state update.

    Wrapped in its own `_node_span`, like every other node, so the batch
    shows up as one parent span in Langfuse — the sub-task spans
    (`build_context`, `delete_context`, ...) execute() triggers underneath
    still appear individually, nested under this one, rather than directly
    under the turn's root span. execute() also emits an `operation_progress`
    custom stream event per task as it completes (see its own docstring) for
    incremental SSE progress finer than "the whole batch just finished"."""
    start = time.perf_counter()
    with _node_span("handle_split_intents") as span:
        result = await execution.execute(state["tasks"], state)
        task_summary = ", ".join(t.type.value for t in state["tasks"])
        entry = _trace(
            "handle_split_intents", None, state["message"],
            f"{len(state['tasks'])} operation(s): {task_summary}", start,
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    result["trace"] = [entry, *result.get("trace", [])]
    return result


async def validate_completeness_node(state: GraphState) -> dict:
    start = time.perf_counter()
    with _node_span("validate_completeness") as span:
        gap_batch = await find_knowledge_gaps(
            state["project_id"], state.get("active_room_id"), state.get("skipped_rooms") or []
        )
        complete = gap_batch is None
        entry = _trace("validate_completeness", None, state["project_id"], f"complete={complete}", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"complete": complete, "trace": [entry]}


async def _build_assumptions(project_id: str) -> list[str]:
    """Human-readable notes on which fields were system-inferred/calculated
    rather than user-stated — built from real provenance
    (app.facts/inference/calculation, Phase 12) rather than a
    manually-accumulated list, unlike the old save_project_node."""
    inferred = await inference.list_inferred(project_id)
    calculated = await calculation.list_calculated(project_id)
    assumptions = []
    for node in inferred + calculated:
        field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type, node.node_type)
        label = FIELD_LABELS.get(field_name, field_name)
        assumptions.append(f"{label}: assumed {node.value}")
    return assumptions


async def complete_project_node(state: GraphState) -> dict:
    """Replaces save_project_node. Just materializes a ProjectContext
    snapshot and generates the closing wrap-up line — find_knowledge_gaps
    returning None already means every structured field has a real value,
    including any room deprioritized into skipped_rooms (see
    question_engine.find_knowledge_gaps' auto-advance — a skipped room is
    revisited, never permanently excluded, so it still has to be answered
    before the project can complete)."""
    start = time.perf_counter()
    with _node_span("complete_project") as span:
        summary = await context_builder.materialize_project_summary(state["project_id"])
        assumptions = await _build_assumptions(state["project_id"])
        project = ProjectContext(session_id=state["session_id"], project_id=state["project_id"], summary=summary, assumptions=assumptions)
        await project.insert()
        entry = _trace(
            "complete_project", None, state["project_id"], f"project_id={project.project_id}, assumptions={len(assumptions)}", start
        )
        span.update(input=entry.input_summary, output=entry.output_summary)

    wrapup_start = time.perf_counter()
    with _node_span("generate_wrapup") as wrapup_span:
        wrapup_capture: dict = {}
        wrapup_message = await llm.generate_wrapup_message(summary, capture=wrapup_capture)
        wrapup_entry = _trace(
            "generate_wrapup", settings.model_question_gen, str(summary), wrapup_message, wrapup_start,
            llm_input=wrapup_capture.get("messages"), llm_output=wrapup_capture.get("raw_output"),
        )
        wrapup_span.update(input=wrapup_entry.input_summary, output=wrapup_entry.output_summary)

    return {"wrapup_message": wrapup_message, "question_generated": True, "trace": [entry, wrapup_entry]}


async def generate_answer_node(state: GraphState) -> dict:
    start = time.perf_counter()
    with _node_span("generate_answer") as span:
        writer = get_stream_writer()
        chunks: list[str] = []
        context = await context_builder.known_fields(state["project_id"], None)
        capture: dict = {}
        async for token in llm.generate_answer(
            state["message"], context, state["history"], state.get("retrieved", ""), capture=capture
        ):
            chunks.append(token)
            writer({"type": "answer_token", "token": token})
        answer = "".join(chunks)
        entry = _trace(
            "generate_answer", settings.model_answer, state["message"], answer, start,
            llm_input=capture.get("messages"), llm_output=answer,
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    # generate_question's own post-edge routes here when needs_answer is set
    # (see _route_post_completeness), and this node's own edge always routes
    # back to generate_question afterward — clearing the flag here is what
    # stops that second pass (a no-op, since question_generated is already
    # True by then) from routing back here again and looping forever.
    return {"answer": answer, "needs_answer": False, "trace": [entry]}


async def generate_question_node(state: GraphState) -> dict:
    """The single consolidated question-generation node — every branch in
    this graph ends up here unless it already produced this turn's trailing
    ask (question_generated). Replaces the old separate generate_question_node
    + analyze_context_node pair (see the pending_gap-centralization plan).

    pending_gap is written by build_context_node right after a value commits,
    and by this node right after it computes the batch it's about to ask
    about — no other node touches it. Normally this node just trusts the
    cached value; it only recomputes via find_knowledge_gaps when there's
    nothing cached yet (the very first gap of a project) or when the cached
    batch's room was JUST added to skipped_rooms by classify_intent_node's
    decline-detection this same turn, which would otherwise make it ask
    about the room the user just declined.

    A batch's every gap shares one room_id (find_knowledge_gaps only ever
    batches one room's fields together), so gap_batch.room_id is written back
    as the new active_room_id — this is what actually persists
    find_knowledge_gaps' auto-advance (moving focus to the next incomplete
    room once the previously active one has nothing left open) back into
    session state; a project-level batch (room_id=None) leaves whatever was
    already active untouched instead of clearing it."""
    start = time.perf_counter()
    with _node_span("generate_question") as span:
        if state.get("question_generated"):
            entry = _trace(
                "generate_question", None, state["project_id"],
                f"question or confirmation already queued: {state.get('pending_question')}", start,
            )
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"trace": [entry]}

        skipped_rooms = state.get("skipped_rooms") or []
        active_room_id = state.get("active_room_id")
        cached = state.get("pending_gap")
        gap_batch = KnowledgeGapBatch(**cached) if cached else None
        if gap_batch is None or (gap_batch.room_id and gap_batch.room_id in skipped_rooms):
            gap_batch = await find_knowledge_gaps(state["project_id"], active_room_id, skipped_rooms)

        if gap_batch is None:
            entry = _trace("generate_question", None, state["project_id"], "complete, no question queued", start)
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"trace": [entry]}

        context = await context_builder.known_fields(state["project_id"], gap_batch.room_id)
        capture: dict = {}
        question = await generate_question(gap_batch.gaps, context, capture=capture)
        entry = _trace(
            "generate_question", settings.model_question_gen,
            ", ".join(g.canonical_path for g in gap_batch.gaps), question, start,
            llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {
        "pending_question": question,
        "pending_gap": gap_batch.model_dump(),
        "current_field": {
            "canonical_paths": [g.canonical_path for g in gap_batch.gaps],
            "room_id": gap_batch.room_id,
        },
        "active_room_id": gap_batch.room_id or active_room_id,
        "question_generated": True,
        "trace": [entry],
    }


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
    """Routes on state["tasks"] (see app.understanding.understand()), not the
    legacy state["intent"] strings — those are kept around only as
    human-readable telemetry (see classify_intent_node)."""

    # classify_intent_node already short-circuited this turn — one or more
    # write tasks still have connection=None (fresh classification or an
    # unanswered/partially-answered resume). Nothing executes this turn;
    # route straight to analyze_context like every other "already asked
    # something" branch.
    if state.get("pending_operation_questions"):
        return "operation_questions_pending"
    tasks = state["tasks"] or [TaskSpec(type=TaskType.ANSWER)]
    task_types = [t.type for t in tasks]
    # Decline no longer short-circuits routing (see classify_intent_node's
    # decline-detection, which just mutates skipped_rooms) — a decline reply
    # is classified and routed normally, same as any other message.
    if len(tasks) > 1:
        return "split"
    return understanding.TASK_TYPE_TO_INTENT[task_types[0]]


def _route_after_delete_context(state: GraphState) -> str:
    # A no-match or ambiguous-match reply sets question_generated (via
    # pending_question) without changing anything, and must skip
    # validate_completeness — routing there instead would let
    # validate_completeness's "incomplete" branch overwrite the
    # clarification question with an unrelated one (that branch also lands
    # on generate_question_node, which DOES guard on question_generated —
    # but only after validate_completeness has already recomputed `complete`
    # against a value that hasn't actually changed). Only an actual
    # retraction reaches validate_completeness.
    return "confirmation_pending" if state.get("question_generated") else "validate"


def _route_completeness(state: GraphState) -> str:
    return "complete" if state["complete"] else "incomplete"


def _route_after_split(state: GraphState) -> str:
    # A split that includes an EDIT_CONTEXT or DELETE_CONTEXT task still needs
    # completeness checked before answering, same as the single-task
    # update_context path — a deletion can re-open a gap just as easily as an
    # edit can close one. A split without either (e.g. RETRIEVE_CONTEXT +
    # DATABASE_QUERY) has nothing new to validate and goes straight to answering.
    task_types = {t.type for t in state["tasks"]}
    return "validate" if task_types & {TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT} else "answer"


def _route_post_completeness(state: GraphState) -> str:
    # Plain single-task update_context turns go straight to generate_question
    # — no separate answer needed there. Only
    # a split turn that also included an info-intent (flagged by
    # handle_split_intents_node) needs a real answer after completeness is
    # resolved.
    return "answer" if state.get("needs_answer") else "analyze"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("query_catalog", query_catalog_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("delete_context", delete_context_node)
    graph.add_node("handle_split_intents", handle_split_intents_node)
    graph.add_node("validate_completeness", validate_completeness_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("complete_project", complete_project_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "CONTEXT_RETRIEVAL": "retrieve_context",
            "DIRECT_ANSWER": "generate_answer",
            "DATABASE_RETRIEVAL": "query_catalog",
            "CONTEXT_UPDATE": "build_context",
            "CONTEXT_DELETE": "delete_context",
            "operation_questions_pending": "generate_question",
            "split": "handle_split_intents",
        },
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("query_catalog", "generate_answer")
    graph.add_edge("generate_answer", "generate_question")



    graph.add_edge("build_context", "validate_completeness")
    # A real retraction (not a no/ambiguous-match reply) still needs
    # completeness re-checked, same as build_context's own edge above: a
    # retraction can re-open a previously-closed slot just as easily as an
    # edit can close one.
    graph.add_conditional_edges(
        "delete_context",
        _route_after_delete_context,
        {"confirmation_pending": "generate_question", "validate": "validate_completeness"},
    )

    graph.add_conditional_edges(
        "handle_split_intents",
        _route_after_split,
        {"validate": "validate_completeness", "answer": "generate_answer"},
    )

    graph.add_conditional_edges(
        "validate_completeness",
        _route_completeness,
        {"complete": "complete_project", "incomplete": "generate_question"},
    )
    graph.add_conditional_edges(
        "complete_project",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": "generate_question"},
    )
    graph.add_conditional_edges(
        "generate_question",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": END},
    )

    return graph.compile()


app_graph = build_graph()
