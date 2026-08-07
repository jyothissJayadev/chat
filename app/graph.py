import asyncio
import operator
import re
import time
from contextlib import contextmanager
from typing import Annotated, Optional, TypedDict

from langfuse import get_client
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz

from app import calculation, canonical_mapper, context_builder, deepinfra, execution, inference, rag, retrieval, understanding, versioning
from app.config import settings
from app.context_builder import ProposedWrite, apply_to_graph, build_context
from app.models import FIELD_LABELS, FIELD_TIERS, RETRY_LIMITS, KnowledgeNode, ProjectContext, TraceEntry
from app.question_engine import KnowledgeGap, find_knowledge_gap, generate_question
from app.tasks import TaskSpec, TaskType

# Re-exported for backward compatibility — these now live in
# app/understanding.py (see its docstring for why), but existing callers
# (tests/test_graph.py) still import them from here.
guard_database_query = understanding.guard_database_query
guard_split_intents = understanding.guard_split_intents


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
    active_room_id: Optional[str]
    # Retry counters for declined questions, keyed by canonical_path — see
    # decline_field_node. Session-level dialogue mechanics (app.models.ChatSession),
    # not a project fact, so it never touches KnowledgeNode.
    field_attempts: dict[str, int]
    intent: list[str]
    tasks: list[TaskSpec]
    retrieved: str
    pending_question: Optional[str]
    # Serialized question_engine.KnowledgeGap for the currently pending
    # question, if any — lets decline_field_node retry without re-deriving
    # the gap (tier, field_label, room_id already on hand).
    pending_gap: Optional[dict]
    # Serialized context_builder.Conflict awaiting a yes/no reply — set by
    # build_context_node when a critical-tier value would change.
    pending_confirmation: Optional[dict]
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
# treated as negative by the caller (build_context_node/confirm_conflict_node)
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


async def classify_intent_node(state: GraphState) -> dict:
    """Thin wrapper around app.understanding.understand() — see
    ARCHITECTURE_BASELINE.md Phase 1. Understanding itself (classification +
    guards) lives in that pure, DB-free module; this node's only jobs are
    threading the LLM capture through into the trace and storing `tasks`,
    which routing reads (Phase 2 cutover)."""
    start = time.perf_counter()
    with _node_span("classify_intent") as span:
        capture: dict = {}
        meaning = await understanding.understand(
            state["message"],
            state["history"],
            pending_field=(state.get("pending_gap") or {}).get("canonical_path"),
            capture=capture,
        )
        # TASK_TYPE_TO_INTENT has no entry for DELETE_CONTEXT (see app.tasks —
        # it's not a synthetic intent string) — this list is telemetry only
        # (_route_intent routes on state["tasks"] directly, never this), so
        # fall back to the enum's own value rather than raising on the one
        # TaskType this map was never meant to cover.
        intent = [understanding.TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in meaning.tasks]

        output_summary = ",".join(intent)
        if intent != meaning.raw_intents:
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
    return {"intent": intent, "tasks": meaning.tasks, "trace": [entry]}


async def retrieve_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    # `task` accepted (and unused) only for signature parity with
    # app.execution.execute()'s uniform dispatch[t.type](state, t) call —
    # RETRIEVE_CONTEXT has no per-clause target/room_hint distinction to make
    # use of (unlike build_context_node/delete_context_node), it always reads
    # the whole turn's message.
    start = time.perf_counter()
    with _node_span("retrieve_context") as span:
        nodes = await retrieval.retrieve_scoped(state["message"], state["project_id"])
        parts = []
        for node in nodes:
            if node.value is None:
                continue
            field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type, node.node_type)
            parts.append(f"{field_name}={node.value}")
        summary = ", ".join(parts) or "no context yet"
        entry = _trace("retrieve_context", None, state["message"], summary, start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"retrieved": summary, "trace": [entry]}


async def query_catalog_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    # See retrieve_context_node's comment above — `task` is accepted only for
    # dispatch signature parity, DATABASE_QUERY has no room-scoped meaning.
    start = time.perf_counter()
    with _node_span("query_catalog") as span:
        items = await rag.query_catalog(state["message"])
        summary = "; ".join(f"{i.title}: {i.description[:80]}" for i in items) or "no matches"
        entry = _trace("query_catalog", settings.model_embedding, state["message"], summary, start)
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
    nodes = await KnowledgeNode.find(KnowledgeNode.project_id == project_id).to_list()
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


async def build_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    """Replaces extract_fields_node + update_context_graph_node: calls
    app.context_builder.build_context(), which writes structured fields
    directly and routes freeform mentions through the canonical mapper (see
    ontology/PHASE7_CONTEXT_BUILDER.md). If the build held back a critical-tier
    value change (BuildResult.pending_confirmations), nothing is silently
    applied — a confirmation question is generated instead and the turn ends
    without checking completeness (see _route_after_build_context).

    `task` is only ever passed by app.execution.execute() (a multi-clause
    turn's per-room segment — see app.understanding.segment_clauses); the
    single-task path routed directly from classify_intent calls this with no
    second argument, so `message`/`active_room_id` below fall back to the
    whole-turn state exactly as before Fix 2 — this function's single-clause
    behavior is unchanged byte-for-byte. `task.room_hint` (a raw room-type
    NAME, not a room_id — see app.tasks.TaskSpec) is passed through
    separately from active_room_id, not merged into it: build_context()
    resolves it via the same fuzzy-match-or-create path as extracted.roomType,
    which a bare id fallback can't go through."""
    start = time.perf_counter()
    message = (task.target if task else None) or state["message"]
    active_room_id = state.get("active_room_id")
    room_hint = task.room_hint if task else None
    with _node_span("build_context") as span:
        result = await build_context(message, state["project_id"], active_room_id, room_hint=room_hint)
        entry = _trace(
            "build_context",
            settings.model_extraction,
            message,
            f"wrote {len(result.written)} node(s), {len(result.pending_confirmations)} pending confirmation(s)",
            start,
        )
        span.update(input=entry.input_summary, output=entry.output_summary)

    if result.pending_confirmations:
        conflict = result.pending_confirmations[0]
        field_name = context_builder.LEAF_TO_FIELD_NAME.get(conflict.node_type, conflict.node_type)
        field_label = FIELD_LABELS.get(field_name, field_name)
        confirm_start = time.perf_counter()
        with _node_span("generate_conflict_confirmation") as cspan:
            capture: dict = {}
            question = await deepinfra.generate_conflict_confirmation(
                field_label, conflict.old_value, conflict.new_value, capture=capture
            )
            confirm_entry = _trace(
                "generate_conflict_confirmation",
                settings.model_question_gen,
                conflict.canonical_path,
                question,
                confirm_start,
                llm_input=capture.get("messages"),
                llm_output=capture.get("raw_output"),
            )
            cspan.update(input=confirm_entry.input_summary, output=confirm_entry.output_summary)
        return {
            "active_room_id": result.room_id,
            "pending_confirmation": {**conflict.model_dump(), "field_label": field_label, "question": question},
            "question_generated": True,
            "trace": [entry, confirm_entry],
        }

    update_summary = await _summarize_written(state["project_id"], result.written)
    return {
        "active_room_id": result.room_id,
        "update_summary": update_summary or None,
        "trace": [entry],
    }


async def confirm_conflict_node(state: GraphState) -> dict:
    """Handles a reply to build_context_node's or delete_context_node's
    held-back conflict (see _route_intent — this only runs when
    pending_confirmation was set last turn). Affirmative applies the new
    value / retracts the node directly (bypassing detect_conflicts — the
    conflict already IS the confirmation); negative or ambiguous discards it
    and leaves the graph untouched. Never mines the reply for additional
    facts, same posture decline_field_node already has for a decline reply.

    Branches on `kind` — "edit" (build_context_node's own conflicts,
    unchanged), "delete" (a single node), "delete_room" (a whole room and
    everything under it, via versioning.retract_subtree)."""
    start = time.perf_counter()
    with _node_span("confirm_conflict") as span:
        conflict = state["pending_confirmation"]
        message = state["message"]
        kind = conflict.get("kind", "edit")
        if _is_affirmative(message):
            if kind in ("delete", "delete_room"):
                node = await KnowledgeNode.find_one(
                    KnowledgeNode.project_id == state["project_id"], KnowledgeNode.canonical_path == conflict["canonical_path"]
                )
                if node is None:
                    outcome = f"confirmed, but {conflict['canonical_path']} no longer exists — nothing to retract"
                elif kind == "delete_room":
                    retracted = await versioning.retract_subtree(node)
                    outcome = f"confirmed: retracted room {conflict['old_value']!r} and {len(retracted) - 1} descendant node(s)"
                else:
                    await versioning.retract_node(node)
                    outcome = f"confirmed: retracted {conflict['old_value']!r}"
            else:
                await apply_to_graph(
                    state["project_id"],
                    [ProposedWrite(canonical_path=conflict["canonical_path"], node_type=conflict["node_type"], value=conflict["new_value"], tier=conflict["tier"])],
                )
                outcome = f"confirmed: applied new value {conflict['new_value']!r}"
        elif _is_negative(message):
            outcome = f"declined: kept {conflict['old_value']!r}"
        else:
            outcome = f"ambiguous reply — kept {conflict['old_value']!r} (never apply an unconfirmed change)"
        entry = _trace("confirm_conflict", None, message, outcome, start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"pending_confirmation": None, "trace": [entry]}


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
    leaves = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.node_type == "RoomType", KnowledgeNode.lifecycle == "active"
    ).to_list()
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
    """Fuzzy-matches a segmented clause's raw room-name hint
    (app.tasks.TaskSpec.room_hint — see app.understanding.segment_clauses)
    against existing LIVE rooms, same confidence bar as
    _match_room_for_deletion's own room-identity check. A hint that matches
    no existing room (e.g. this is the first mention of that room) simply
    resolves to None — the caller's own active_room_id fallback takes over,
    same as having no hint at all."""
    if not room_hint:
        return None
    rooms = await _existing_room_map(project_id)
    best_room_id, best_score = None, 0
    for room_id, room_type in rooms.items():
        score = fuzz.ratio(room_hint.strip().lower(), room_type.lower())
        if score > best_score:
            best_room_id, best_score = room_id, score
    return best_room_id if best_score >= _ROOM_DELETE_MATCH_THRESHOLD else None


async def _handle_room_deletion(state: GraphState, room_id: str, start: float, span) -> dict:
    """A room is always critical tier — this always holds for confirmation
    (see 1.9's routing note: a retraction can re-open a slot, so this and
    the item-level branch both route through validate_completeness rather
    than straight to analyze_context). The actual cascade
    (versioning.retract_subtree) only runs once confirm_conflict_node sees
    an affirmative reply next turn — nothing is touched here yet."""
    room_node = await KnowledgeNode.find_one(
        KnowledgeNode.project_id == state["project_id"], KnowledgeNode.canonical_path == f"Project.Rooms.{room_id}"
    )
    if room_node is None:
        entry = _trace("delete_context", None, room_id, "room match had no container node — skipped", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
        return {"trace": [entry]}

    room_type_node = await KnowledgeNode.find_one(
        KnowledgeNode.project_id == state["project_id"], KnowledgeNode.canonical_path == f"Project.Rooms.{room_id}.RoomType"
    )
    room_label = str(room_type_node.value) if room_type_node and room_type_node.value else f"room {room_id}"

    question = await deepinfra.generate_conflict_confirmation_delete(room_label, capture={})
    entry = _trace("delete_context", None, room_id, f"room deletion held for confirmation: {room_label}", start)
    span.update(input=entry.input_summary, output=entry.output_summary)
    return {
        "pending_confirmation": {
            "kind": "delete_room", "canonical_path": room_node.canonical_path, "node_type": "Rooms",
            # field_label is required by app.chat.run_chat_turn's confirm_change
            # SSE event (build_context_node's own conflict dict always carries
            # one — see its "field_label" key below) — a room deletion has no
            # FIELD_LABELS entry to draw from (it's not a structured slot
            # field), so "room" is the generic, always-correct description.
            "field_label": "room",
            "old_value": room_label, "new_value": None, "tier": "critical", "question": question,
        },
        "question_generated": True,
        "trace": [entry],
    }


async def delete_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    """The write path for a DELETE_CONTEXT task (see app.understanding.guard_delete).
    Checks for a room-level deletion first (see _match_room_for_deletion),
    then falls through to item-level matching among live freeform facts
    (canonical_mapper.resolve_deletion_target) — never fabricates a match: no
    confident match asks for clarification, an ambiguous one lists the
    candidates instead of guessing, and a critical-tier item routes through
    the same confirm_conflict machinery build_context_node already uses for
    value changes rather than applying immediately. `task.room_hint` is a raw
    room-type NAME (app.tasks.TaskSpec), not a room_id — resolved here via
    _resolve_room_hint's fuzzy match against existing rooms, falling back to
    the session's active_room_id exactly like the single-task path already
    did before Fix 2."""
    start = time.perf_counter()
    target_text = (task.target if task else None) or state["message"]
    room_id = await _resolve_room_hint(state["project_id"], task.room_hint if task else None) or state.get("active_room_id")

    with _node_span("delete_context") as span:
        room_match = await _match_room_for_deletion(state["project_id"], target_text)
        if room_match is not None:
            return await _handle_room_deletion(state, room_match, start, span)

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
        # LEAF_TO_FIELD_NAME only covers structured slot fields (ProjectType,
        # Budget, ...) — every freeform node_type this function can ever match
        # (Materials/Furniture/Attributes/Constraints/ClientPreferences) has no
        # entry, so field_tier is always "moderate" here today: none of those
        # types is a critical-tier project slot, so immediate application is
        # the correct default, not a gap. The branch below still exists so a
        # future ontology addition doesn't silently skip confirmation.
        field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type)
        field_tier = FIELD_TIERS.get(field_name, "moderate") if field_name else "moderate"

        if field_tier == "critical":
            question = await deepinfra.generate_conflict_confirmation_delete(label, capture={})
            entry = _trace("delete_context", None, target_text, f"critical deletion held for confirmation: {node.canonical_path}", start)
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {
                "pending_confirmation": {
                    "kind": "delete", "canonical_path": node.canonical_path, "node_type": node.node_type,
                    # field_label is required by app.chat.run_chat_turn's
                    # confirm_change SSE event — see _handle_room_deletion's
                    # matching comment; FIELD_LABELS has no entry for a
                    # freeform node_type, so fall back to the node_type itself.
                    "field_label": FIELD_LABELS.get(field_name) or node.node_type.lower(),
                    "old_value": label, "new_value": None, "tier": field_tier, "question": question,
                },
                "question_generated": True,
                "trace": [entry],
            }

        await versioning.retract_node(node)
        entry = _trace("delete_context", None, target_text, f"retracted {node.canonical_path}", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
        return {"update_summary": f"Removed {label!r}.", "trace": [entry]}


async def handle_split_intents_node(state: GraphState) -> dict:
    """Only reached when classify_intent found more than one distinct ask in
    the message (see _route_intent). Delegates to app.execution.execute(),
    which runs each requested operation concurrently via asyncio.gather — not
    LangGraph fan-out — then merges results back into a single partial state
    update."""
    return await execution.execute(state["tasks"], state)


async def decline_field_node(state: GraphState) -> dict:
    """Handles an explicit decline of the currently pending question: rephrase
    (softer wording, same field) while under the field's tier retry budget,
    or infer a value immediately once exhausted (app.inference.infer_field) —
    closing the gap right away rather than deferring to a later sweep, so
    find_knowledge_gap naturally moves past it next call."""
    start = time.perf_counter()
    with _node_span("decline_field") as span:
        gap_dict = state["pending_gap"]
        canonical_path = gap_dict["canonical_path"]
        tier = gap_dict["tier"]
        limit = RETRY_LIMITS.get(tier, 0)
        attempts = state["field_attempts"].get(canonical_path, 0) + 1
        field_attempts = {**state["field_attempts"], canonical_path: attempts}
        room_id = gap_dict.get("room_id") or state.get("active_room_id")

        if attempts <= limit:
            context = await context_builder.known_fields(state["project_id"], room_id)
            capture: dict = {}
            question = await generate_question(KnowledgeGap(**gap_dict), context, is_retry=True, capture=capture)
            entry = _trace(
                "decline_field",
                settings.model_question_gen,
                canonical_path,
                f"rephrase {attempts}/{limit}: {question}",
                start,
                llm_input=capture.get("messages"),
                llm_output=capture.get("raw_output"),
            )
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {
                "field_attempts": field_attempts,
                "pending_question": question,
                "pending_gap": gap_dict,
                "question_generated": True,
                "trace": [entry],
            }

        context = await context_builder.known_fields(state["project_id"], room_id)
        await inference.infer_field(state["project_id"], canonical_path, gap_dict["node_type"], gap_dict["field_label"], context, room_id=room_id)
        entry = _trace("decline_field", None, canonical_path, f"exhausted after {attempts} attempts, inferred a value", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
        return {
            "field_attempts": field_attempts,
            "pending_question": None,
            "pending_gap": None,
            "trace": [entry],
        }


async def validate_completeness_node(state: GraphState) -> dict:
    start = time.perf_counter()
    with _node_span("validate_completeness") as span:
        gap = await find_knowledge_gap(state["project_id"], state.get("active_room_id"))
        complete = gap is None
        entry = _trace("validate_completeness", None, state["project_id"], f"complete={complete}", start)
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {"complete": complete, "trace": [entry]}


async def generate_question_node(state: GraphState) -> dict:
    start = time.perf_counter()
    with _node_span("generate_question") as span:
        # validate_completeness only routes here when a gap exists, so this
        # is guaranteed to find one.
        gap = await find_knowledge_gap(state["project_id"], state.get("active_room_id"))
        context = await context_builder.known_fields(state["project_id"], gap.room_id or state.get("active_room_id"))
        capture: dict = {}
        question = await generate_question(gap, context, capture=capture)
        entry = _trace(
            "generate_question", settings.model_question_gen, gap.canonical_path, question, start,
            llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {
        "pending_question": question,
        "pending_gap": gap.model_dump(),
        "question_generated": True,
        "trace": [entry],
    }


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
    """Replaces save_project_node. No inference sweep here — decline_field_node
    already infers a value the moment a field's retry budget is exhausted, so
    every structured field already has a real value by the time
    find_knowledge_gap returns None. Just materializes a ProjectContext
    snapshot and generates the closing wrap-up line."""
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
        wrapup_message = await deepinfra.generate_wrapup_message(summary, capture=wrapup_capture)
        wrapup_entry = _trace(
            "generate_wrapup", settings.model_question_gen, str(summary), wrapup_message, wrapup_start,
            llm_input=wrapup_capture.get("messages"), llm_output=wrapup_capture.get("raw_output"),
        )
        wrapup_span.update(input=wrapup_entry.input_summary, output=wrapup_entry.output_summary)

    return {"wrapup_message": wrapup_message, "question_generated": True, "trace": [entry, wrapup_entry]}


async def generate_answer_node(state: GraphState) -> dict:
    async def _answer() -> tuple[str, TraceEntry]:
        start = time.perf_counter()
        with _node_span("generate_answer") as span:
            writer = get_stream_writer()
            chunks: list[str] = []
            context = await context_builder.known_fields(state["project_id"], state.get("active_room_id"))
            capture: dict = {}
            async for token in deepinfra.generate_answer(
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
        return answer, entry

    async def _question() -> tuple[str | None, dict | None, TraceEntry | None]:
        # Safe to run concurrently with the answer call — see the matching
        # comment history in this function pre-cutover. question_generated
        # is already True whenever a pending_confirmation was set earlier
        # this turn, so this naturally skips generating a redundant question.
        if state.get("question_generated"):
            return state.get("pending_question"), state.get("pending_gap"), None
        start = time.perf_counter()
        gap = await find_knowledge_gap(state["project_id"], state.get("active_room_id"))
        if gap is None:
            return None, None, None
        with _node_span("generate_question") as span:
            context = await context_builder.known_fields(state["project_id"], gap.room_id or state.get("active_room_id"))
            capture: dict = {}
            question = await generate_question(gap, context, capture=capture)
            entry = _trace(
                "generate_question", settings.model_question_gen, gap.canonical_path, question, start,
                llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
            )
            span.update(input=entry.input_summary, output=entry.output_summary)
        return question, gap.model_dump(), entry

    (answer, answer_entry), (question, gap_dict, question_entry) = await asyncio.gather(_answer(), _question())

    result: dict = {"answer": answer, "trace": [answer_entry]}
    if question_entry is not None:
        result["pending_question"] = question
        result["pending_gap"] = gap_dict
        result["question_generated"] = True
        result["trace"].append(question_entry)
    return result


async def analyze_context_node(state: GraphState) -> dict:
    """Runs after every branch. Queues the next knowledge-gap question
    regardless of which branch just ran, so an unrelated question mid-flow
    doesn't drop it — unless a question or confirmation was already queued
    this turn (question_generated), or the project is already complete."""
    start = time.perf_counter()
    with _node_span("analyze_context") as span:
        if state.get("question_generated"):
            entry = _trace("analyze_context", None, state["project_id"], f"question or confirmation already queued: {state.get('pending_question')}", start)
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"trace": [entry]}

        gap = await find_knowledge_gap(state["project_id"], state.get("active_room_id"))
        if gap is None:
            entry = _trace("analyze_context", None, state["project_id"], "complete, no question queued", start)
            span.update(input=entry.input_summary, output=entry.output_summary)
            return {"trace": [entry]}

        context = await context_builder.known_fields(state["project_id"], gap.room_id or state.get("active_room_id"))
        capture: dict = {}
        question = await generate_question(gap, context, capture=capture)
        entry = _trace(
            "analyze_context", settings.model_question_gen, gap.canonical_path, question, start,
            llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)
    return {
        "pending_question": question,
        "pending_gap": gap.model_dump(),
        "question_generated": True,
        "trace": [entry],
    }


async def describe_image_node(state: GraphState, image_url: str) -> tuple[TraceEntry, str]:
    start = time.perf_counter()
    with _node_span("describe_image") as span:
        capture: dict = {}
        description = await deepinfra.vision(image_url, state["message"] or "Describe this room.", capture=capture)
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
    if state.get("pending_confirmation"):
        return "confirm_conflict"
    tasks = state["tasks"] or [TaskSpec(type=TaskType.ANSWER)]
    task_types = [t.type for t in tasks]
    if (
        TaskType.EDIT_CONTEXT not in task_types
        and TaskType.DELETE_CONTEXT not in task_types
        and state.get("pending_gap")
        and _is_decline(state["message"])
    ):
        return "decline_field"
    if len(tasks) > 1:
        return "split"
    # DELETE_CONTEXT has no entry in TASK_TYPE_TO_INTENT (see app.tasks —
    # it's not a synthetic intent string, guard_delete produces it directly
    # from EDIT_CONTEXT) — routed here explicitly instead.
    if task_types[0] == TaskType.DELETE_CONTEXT:
        return "delete_context"
    return understanding.TASK_TYPE_TO_INTENT[task_types[0]]


def _route_decline(state: GraphState) -> str:
    # A rephrase leaves pending_question set (decline_field_node already
    # produced the next question); an exhausted field leaves it None and
    # needs the normal completeness check to decide what's next.
    return "rephrasing" if state.get("pending_question") else "continue"


def _route_after_build_context(state: GraphState) -> str:
    # A held-back conflict means nothing new was actually applied this turn
    # — skip straight to analyze_context (which no-ops, since
    # question_generated is already True) rather than checking completeness
    # against a value that hasn't changed.
    return "confirmation_pending" if state.get("pending_confirmation") else "validate"


def _route_after_delete_context(state: GraphState) -> str:
    # Mirrors _route_after_build_context above, generalized to
    # delete_context_node's extra outcomes: a no-match or ambiguous-match
    # reply also sets question_generated (via pending_question, not
    # pending_confirmation) without changing anything, and must skip
    # validate_completeness the same way a held-back conflict does — routing
    # there instead would let generate_question_node's "incomplete" branch
    # overwrite the clarification question with an unrelated one, since that
    # node has no question_generated guard of its own (only analyze_context
    # does). Only an actual retraction reaches validate_completeness.
    return "confirmation_pending" if state.get("question_generated") else "validate"


def _route_completeness(state: GraphState) -> str:
    return "complete" if state["complete"] else "incomplete"


def _route_after_split(state: GraphState) -> str:
    if state.get("pending_confirmation"):
        return "answer" if state.get("needs_answer") else "confirmation_only"
    # A split that includes an EDIT_CONTEXT or DELETE_CONTEXT task still needs
    # completeness checked before answering, same as the single-task
    # update_context path — a deletion can re-open a gap just as easily as an
    # edit can close one. A split without either (e.g. RETRIEVE_CONTEXT +
    # DATABASE_QUERY) has nothing new to validate and goes straight to answering.
    task_types = {t.type for t in state["tasks"]}
    return "validate" if task_types & {TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT} else "answer"


def _route_post_completeness(state: GraphState) -> str:
    # Plain single-task update_context turns, and decline_field/confirm_conflict
    # turns, go straight to analyze_context — no separate answer needed
    # there. Only a split turn that also included an info-intent (flagged by
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
    graph.add_node("confirm_conflict", confirm_conflict_node)
    graph.add_node("decline_field", decline_field_node)
    graph.add_node("handle_split_intents", handle_split_intents_node)
    graph.add_node("validate_completeness", validate_completeness_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("complete_project", complete_project_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("analyze_context", analyze_context_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "context_related": "retrieve_context",
            "direct_question": "generate_answer",
            "database_query": "query_catalog",
            "update_context": "build_context",
            "delete_context": "delete_context",
            "decline_field": "decline_field",
            "confirm_conflict": "confirm_conflict",
            "split": "handle_split_intents",
        },
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("query_catalog", "generate_answer")
    graph.add_edge("generate_answer", "analyze_context")

    graph.add_conditional_edges(
        "decline_field",
        _route_decline,
        {"rephrasing": "analyze_context", "continue": "validate_completeness"},
    )
    graph.add_edge("confirm_conflict", "validate_completeness")

    graph.add_conditional_edges(
        "build_context",
        _route_after_build_context,
        {"confirmation_pending": "analyze_context", "validate": "validate_completeness"},
    )
    # A real retraction (not a no/ambiguous-match reply or a held-back
    # critical-tier confirmation — see _route_after_delete_context) still
    # needs completeness re-checked, same as build_context's own "validate"
    # branch: a retraction can re-open a previously-closed slot just as
    # easily as an edit can close one.
    graph.add_conditional_edges(
        "delete_context",
        _route_after_delete_context,
        {"confirmation_pending": "analyze_context", "validate": "validate_completeness"},
    )

    graph.add_conditional_edges(
        "handle_split_intents",
        _route_after_split,
        {"validate": "validate_completeness", "answer": "generate_answer", "confirmation_only": "analyze_context"},
    )

    graph.add_conditional_edges(
        "validate_completeness",
        _route_completeness,
        {"complete": "complete_project", "incomplete": "generate_question"},
    )
    graph.add_conditional_edges(
        "complete_project",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": "analyze_context"},
    )
    graph.add_conditional_edges(
        "generate_question",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": "analyze_context"},
    )

    graph.add_edge("analyze_context", END)

    return graph.compile()


app_graph = build_graph()
