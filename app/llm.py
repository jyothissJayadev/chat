import json
import logging
from collections.abc import AsyncIterator
from typing import Literal, Optional

import instructor
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, Field

from app import observability  # noqa: F401  (constructs the Langfuse singleton before AsyncOpenAI below)
from app import prompts
from app.config import settings
from langfuse.openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
structured_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

# Fireworks-specific request params for gpt-oss-120b (every text role below
# except vision/embed — see config.py, "reasoning model" caution). Passed via
# extra_body since neither is a standard OpenAI chat-completions field the
# openai SDK exposes as a named kwarg. "low" keeps hidden reasoning-token
# generation down for latency, same posture as this module's existing
# fast/low-stakes model choices; "disabled" turns off reasoning-history
# prompt formatting — this module never reads or re-feeds reasoning/analysis
# content back into context anywhere (every call site below only ever uses
# `.content`), so there's nothing here that depended on it being on.
_REASONING_KWARGS = {"reasoning_effort": "low", "reasoning_history": "disabled"}


class MaterialSpec(BaseModel):
    """Material choice for one specific item/surface, e.g. flooring, countertop,
    sofa — extract_fields' structured-output shape for one ExtractedFields.materials
    entry. Used to live in app/models.py as a storage type (RoomContext.materials);
    now purely an extraction-schema type since the KnowledgeNode cutover stores
    each material as its own Label/Material/Specification leaves instead
    (see app/context_builder.py)."""

    item: str
    material: str
    specification: Optional[str] = None


# extract_graph_links' structured-output relation vocabulary. Used to live in
# app/models.py as ContextGraph's storage-level relation type; now purely an
# extraction-schema constraint — app/context_builder.py doesn't persist these
# as KnowledgeEdge rows yet (see its module docstring), it just surfaces them
# in BuildResult.freeform_relationships. Deliberately still includes
# part_of/located_in (containment relations the LLM's prompt below still
# describes) even though KnowledgeEdgeRelation (app/models.py, Phase 9) drops
# both as superseded by canonical_path — this is what the MODEL is allowed to
# say, not what gets stored.
GraphRelation = Literal[
    "part_of",
    "located_in",
    "uses_material",
    "applies_to",
    "modifies",
    "requires",
    "budget_for",
    "rejected_in_favor_of",
    "revises",
]


def _duplicate_json_keys(raw_json_text: str) -> list[str]:
    """Standard JSON parsing silently keeps only the LAST value for a
    repeated key — live-observed on this model: given a message stating both
    a project-wide total and a room-specific budget, it sometimes emits
    'budgetOrRequirement' twice (once per figure) instead of routing the
    total to overallBudget, and the first figure vanishes with no error, no
    exception, nothing to salvage against. This can't be fixed after the
    fact (which of two values was 'right' isn't recoverable once one is
    already gone) — the caller logs the collision so it's visible instead of
    silent; see ExtractedFields.overallBudget/budgetOrRequirement's
    descriptions for the actual prompt-level fix."""
    duplicates: list[str] = []

    def hook(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    try:
        json.loads(raw_json_text, object_pairs_hook=hook)
    except ValueError:
        pass
    return duplicates


OperationIntent = Literal[
    "CONTEXT_UPDATE", "CONTEXT_DELETE", "CONTEXT_RETRIEVAL", "DATABASE_RETRIEVAL", "DIRECT_ANSWER"
]


class Operation(BaseModel):
    text: str = Field(description="the original meaningful operation text, preserving the user's wording")
    intent: OperationIntent
    # Grounds this operation to a spot in the project tree, per the
    # project-state text classify_operations_user() feeds the model — a
    # root-relative canonical path ("Rooms.a1b2c3d4.Materials.countertop"),
    # a not-yet-existing entity name ("Living Room"), or None when the model
    # can't confidently place it. app.graph's clarifying-question branch
    # asks the user directly for every operation where this is None.
    connection: Optional[str] = None
    # Assigned by classify_operations() after parsing, by list order
    # ("op_1", "op_2", ...) — never requested from the model itself, so a
    # duplicate/missing id can never happen.
    id: str = ""


class OperationClassification(BaseModel):
    operations: list[Operation] = Field(default_factory=list)


# Replaces classify_intent (below) — one structured call that both splits the
# message into independent operations AND labels each with one intent,
# instead of a single label (rarely two, via comma-splitting) per whole
# message plus separate regex guards/segmentation layered on top in
# app/understanding.py. Prompt text (with its own rationale) lives in
# app/prompts.py — CLASSIFY_OPERATIONS_SYSTEM.


async def _salvage_operations(exc: InstructorRetryException, capture: dict | None = None) -> Optional[list[Operation]]:
    """Same recovery strategy as _salvage_extracted_fields/_salvage_extracted_graph_links —
    scans every failed attempt for a real tool call or a JSON object embedded in
    plain text content, so a parse hiccup doesn't silently drop an entire
    turn's worth of operations."""
    for attempt in exc.failed_attempts or []:
        completion = getattr(attempt, "completion", None)
        if not completion or not completion.choices:
            continue
        message = completion.choices[0].message

        raw: Optional[str] = None
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            raw = tool_calls[0].function.arguments
        elif message.content:
            content = message.content
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = content[start : end + 1]

        if not raw:
            continue
        try:
            payload = json.loads(raw)
            result = OperationClassification.model_validate(payload)
        except (ValueError, TypeError):
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result.operations
    return None


async def classify_operations(
    message: str,
    history: str = "",
    pending_field: str | None = None,
    *,
    tree_text: str = "Project",
    capture: dict | None = None,
) -> list[Operation]:
    """Wraps _classify_operations_retrying with op.id assignment — every
    caller gets ids ("op_1", "op_2", ...) by list order, never requested from
    the model itself (see Operation.id's docstring). `tree_text` grounds
    each operation's `connection` against the project's current state (see
    app.context_builder.render_project_tree_text) — defaults to a bare,
    empty-project tree so existing callers that don't pass it (mostly tests)
    keep working."""
    operations = await _classify_operations_retrying(message, history, pending_field, tree_text, capture=capture)
    for i, op in enumerate(operations, start=1):
        op.id = f"op_{i}"
    return operations


async def _classify_operations_retrying(
    message: str,
    history: str,
    pending_field: str | None,
    tree_text: str,
    *,
    capture: dict | None = None,
) -> list[Operation]:
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            return await _classify_operations_raw(message, history, pending_field, tree_text, max_retries, capture=capture)
        except InstructorRetryException as exc:
            salvaged = await _salvage_operations(exc, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _classify_operations_raw(
    message: str,
    history: str,
    pending_field: str | None,
    tree_text: str,
    max_retries: int,
    capture: dict | None = None,
) -> list[Operation]:
    messages = [
        {"role": "system", "content": prompts.classify_operations_system(tree_text)},
        {"role": "user", "content": prompts.classify_operations_user(message, history, pending_field)},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_intent_classifier,
        response_model=OperationClassification,
        max_tokens=1024,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.operations


class RoomResolutionOption(BaseModel):
    # None for an inferred (not-yet-existing) room — see
    # app.prompts.ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE rule 11. A real
    # existing room always carries its exact room id (rule 10).
    id: Optional[str] = None
    label: str


class RoomResolutionItem(BaseModel):
    """One operation's result from resolve_room_connections() — see
    app.prompts.ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE. Always carries a
    question and at least one option, even when the room is confidently
    resolved (rule 15: a single option then stands in for "confirm this
    one") — there is no "resolvable without asking" verdict in this prompt,
    unlike the old generate_clarification_question/ClarificationQuestion it
    replaces. That's what keeps app.graph.classify_intent_node's
    always-block posture (see the classifier-connection-always-block memory)
    a property of the prompt itself rather than something the caller has to
    enforce by ignoring a confidence flag."""

    text: str
    intent: str
    question: str
    options: list[RoomResolutionOption] = Field(default_factory=list)


class RoomResolutionBatch(BaseModel):
    resolutions: list[RoomResolutionItem] = Field(default_factory=list)


async def _salvage_room_resolutions(
    exc: InstructorRetryException, capture: dict | None = None
) -> Optional[list[RoomResolutionItem]]:
    """Same recovery strategy as _salvage_operations — scans every failed
    attempt for a real tool call or a JSON array embedded in plain text
    content, so a parse hiccup doesn't silently drop the whole batch's worth
    of room questions."""
    for attempt in exc.failed_attempts or []:
        completion = getattr(attempt, "completion", None)
        if not completion or not completion.choices:
            continue
        message = completion.choices[0].message

        raw: Optional[str] = None
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            raw = tool_calls[0].function.arguments
        elif message.content:
            content = message.content
            start, end = content.find("["), content.rfind("]")
            if start != -1 and end != -1 and end > start:
                raw = content[start : end + 1]

        if not raw:
            continue
        try:
            payload = json.loads(raw)
            items = payload["resolutions"] if isinstance(payload, dict) else payload
            result = [RoomResolutionItem.model_validate(item) for item in items]
        except (ValueError, TypeError, KeyError):
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result
    return None


async def resolve_room_connections(
    tree_text: str,
    operations: list[dict],
    *,
    capture: dict | None = None,
) -> list[RoomResolutionItem]:
    """Every write operation in this turn whose connection came back None
    from classify_operations, resolved in ONE batched call — replaces the
    old generate_clarification_question, which ran once per unresolved
    operation via asyncio.gather. `operations` is
    `[{"text", "intent", "connection": None}, ...]`, in the exact order
    app.graph._generate_operation_questions wants results back in (the
    prompt's own rule 1 preserves input order, and rule 15 guarantees one
    result per input operation — no id round-trip needed, the caller matches
    positionally)."""
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            return await _resolve_room_connections_raw(tree_text, operations, max_retries, capture=capture)
        except InstructorRetryException as exc:
            salvaged = await _salvage_room_resolutions(exc, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _resolve_room_connections_raw(
    tree_text: str,
    operations: list[dict],
    max_retries: int,
    capture: dict | None = None,
) -> list[RoomResolutionItem]:
    operations_json = json.dumps(operations, indent=2)
    messages = [
        {"role": "system", "content": prompts.room_resolution_agent_system(tree_text, operations_json)},
        {"role": "user", "content": prompts.room_resolution_agent_user()},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_intent_classifier,
        response_model=RoomResolutionBatch,
        max_tokens=1024,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.resolutions


class AdditionalRoomBudget(BaseModel):
    """A budget/requirement figure for a room OTHER than the one already
    detailed in roomType/budgetOrRequirement this turn. Exists because a
    single message can state a figure for a second room the extraction
    schema otherwise has no slot for — without this, the model either
    duplicated a JSON key or invented a nonexistent field name trying to
    represent it (both live-observed via Langfuse), corrupting or silently
    losing that room's figure."""

    roomType: str = Field(description="the OTHER room this figure is for, e.g. 'kitchen' — user's own words are fine")
    budgetOrRequirement: str = Field(description="that room's own figure, e.g. '4 lakh' or '$8k'")


class ExtractedFields(BaseModel):
    """One turn's worth of extraction: project-level fields plus the fields of
    whichever single room the message was about, if any. app/graph.py decides
    which RoomContext this applies to (matching on roomType, or starting a new
    room) — this model itself has no notion of which room it's for.

    Field-specific guidance lives in each Field's description (part of the
    tool-call JSON schema instructor sends), not in the system prompt — for
    the small Turbo model used here, a longer system prompt measurably raised
    the rate of it drifting into replying with a literal '<function=...>'
    text block instead of a real tool call, which instructor's TOOLS mode
    then can't parse at all. Keep the system prompt short; put detail here."""

    projectType: Optional[str] = Field(None, description="new construction, renovation, or a partial refresh/update")
    overallBudget: Optional[str] = Field(
        None,
        description=(
            "ONLY the TOTAL/whole-project budget across every room combined, "
            "e.g. 'the total budget is 15 lakh' -> '15 lakh'. Never a single "
            "room's own figure, even if that room happens to be the one named "
            "in roomType — a figure tied to one specific room (named or not) "
            "always belongs in budgetOrRequirement or additionalRoomBudgets "
            "instead, never here."
        ),
    )
    timeline: Optional[str] = None
    moreRoomsPending: Optional[bool] = Field(
        None, description="true/false ONLY if the user directly answered whether other rooms are in scope"
    )
    roomType: Optional[str] = Field(None, description="e.g. kitchen, living room, bedroom — user's own words are fine")
    budgetOrRequirement: Optional[str] = Field(
        None,
        description=(
            "budget or need for THIS one room (the roomType above) "
            "specifically, e.g. '$8k' or a plain description — never the "
            "whole-project total (that's overallBudget) and never a figure "
            "that actually belongs to a DIFFERENT room (that's "
            "additionalRoomBudgets). If the message gives a room name no "
            "figure of its own, or two different figures for two different "
            "rooms, pick whichever room has the most detail as roomType/"
            "budgetOrRequirement and route the other room's own figure to "
            "additionalRoomBudgets — never invent a new field name and never "
            "let two different figures collide into this one field."
        ),
    )
    additionalRoomBudgets: Optional[list[AdditionalRoomBudget]] = Field(
        None,
        description=(
            "A budget/requirement figure stated for a room OTHER than the one "
            "already detailed in roomType/budgetOrRequirement — e.g. roomType "
            "is 'living room' but the message ALSO gives a figure for "
            "'kitchen': put {roomType: 'kitchen', budgetOrRequirement: '4 "
            "lakh'} here. One entry per additional room that has its own "
            "figure. Do not repeat the primary room here, and do not repeat "
            "these rooms in mentionedAdditionalRooms.\n"
            "Worked example: \"total's around 15 lakh, maybe 4 for the "
            "kitchen, still deciding on the bedroom\" with roomType left as "
            "whatever room the message is actually centered on -> "
            "overallBudget='around 15 lakh' (hedge kept, never nulled), "
            "additionalRoomBudgets=[{roomType: 'kitchen', "
            "budgetOrRequirement: 'maybe 4 lakh'}], and 'bedroom' goes in "
            "mentionedAdditionalRooms since it has no figure of its own yet — "
            "never invent one for it."
        ),
    )
    style: Optional[str] = None
    squareFootage: Optional[float] = None
    existingFurniture: Optional[str] = None
    materials: Optional[list[MaterialSpec]] = Field(
        None, description="items newly mentioned this message only, e.g. {item: 'flooring', material: 'oak wood'}"
    )
    mentionedAdditionalRooms: Optional[list[str]] = Field(
        None,
        description=(
            "Other room types mentioned in this message that have NO figure "
            "of their own (a room WITH its own figure goes in "
            "additionalRoomBudgets instead, not here) besides the one already "
            "being detailed above (roomType/budgetOrRequirement/style/etc.) — "
            "names only, e.g. ['bedroom', 'kitchen']. Use this when the user "
            "lists several rooms in one message but only gives detailed facts "
            "about one of them; the others get asked about individually on a "
            "later turn. Do not repeat the room already covered by roomType here."
        ),
    )


def _raw_completion_text(completion) -> str:
    """Best-effort raw text of a chat completion — the tool-call arguments
    JSON when the model made a real tool call (TOOLS mode structured output),
    else plain message content."""
    try:
        message = completion.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls[0].function.arguments
    return message.content or ""


def _salvage_extracted_fields(
    exc: InstructorRetryException, message: str, capture: dict | None = None
) -> Optional[ExtractedFields]:
    """Recovers the answer instructor's strict TOOLS-mode parser discarded.

    Live-observed: this model's FIRST attempt almost always arrives correctly
    filled in but as its own native '<function=Name>{...}</function>' text
    syntax (see the comment below) instead of a real tool call, which TOOLS
    mode rejects outright — and every retry after that degrades to a fully
    empty completion rather than recovering (also live-observed, consistently
    across repeated failures), so without this, a turn's real answer was
    being thrown away and silently dropped instead of saved. Scans every
    failed attempt's raw text for a JSON object and validates it against the
    real schema; if nothing parses, the caller's caller falls back to
    extract_fields_node's existing skip-this-turn handling."""
    for attempt in exc.failed_attempts or []:
        completion = getattr(attempt, "completion", None)
        content = completion.choices[0].message.content if completion and completion.choices else None
        if not content:
            continue
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        raw = content[start : end + 1]
        duplicates = _duplicate_json_keys(raw)
        if duplicates:
            logger.warning(
                "extract_fields salvage: duplicate key(s) %s in model output for message %r — "
                "only the LAST value survives JSON parsing, earlier value(s) silently lost. raw=%s",
                duplicates, message[:200], raw[:500],
            )
        try:
            result = ExtractedFields.model_validate_json(raw)
        except ValueError:
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result
    return None


async def extract_fields(message: str, known: dict, *, capture: dict | None = None) -> ExtractedFields:
    # Filter out unset fields before formatting into the prompt. Showing a dict
    # with every field explicitly set to None (e.g. "Known so far: {'roomType':
    # None, ...}") reliably confused a non-reasoning model into extracting
    # nothing at all — reproducibly, not flaky — even though the exact same
    # message alone extracts correctly. Only showing fields that are actually
    # known is unambiguous for any model.
    known = {k: v for k, v in known.items() if v is not None}

    # Two-stage retry strategy, confirmed by a live Langfuse trace: this
    # model's first attempt almost always already has the right answer, just
    # wrapped in its own '<function=...>{...}</function>' text syntax instead
    # of a real tool call — instructor's TOOLS mode sees no populated
    # tool_calls field and treats that as "no call made" regardless of what
    # the text contains, then retries. But every retry after the first tends
    # to degrade into a near-empty completion instead of recovering (also
    # live-observed — see the trace: attempt 1 returned 70 output tokens,
    # attempts 2-9 returned 1-4 each), so spending the full retry budget
    # before ever checking whether attempt 1 was salvageable wastes most of a
    # turn's latency (12s of a 20s turn, observed) for no benefit. Try a
    # single attempt first and salvage its raw text immediately on failure;
    # only fall back to the full remaining retry budget — genuinely rare,
    # attempt 1 not even salvageable — if that comes up empty. Same total
    # attempt ceiling either way (1 + 8 = 9, same as the previous flat
    # max_retries=8).
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 7):
        try:
            return await _extract_fields_raw(message, known, max_retries, capture=capture)
        except InstructorRetryException as exc:
            salvaged = _salvage_extracted_fields(exc, message, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _extract_fields_raw(
    message: str, known: dict, max_retries: int, capture: dict | None = None
) -> ExtractedFields:
    messages = [
        {"role": "system", "content": prompts.EXTRACT_FIELDS_SYSTEM},
        {"role": "user", "content": prompts.extract_fields_user(known, message)},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_extraction,
        response_model=ExtractedFields,
        # Without a cap, a message with nothing to extract (e.g. a question
        # like "what is the cost for kitchen" instead of new project info) was
        # observed to send generation into a multi-minute stall in tool-calling
        # mode — the same class of unbounded-output issue as generate_question
        # and extract_graph_links, just never hit here until now. A small JSON
        # object never needs anywhere near this many tokens.
        max_tokens=1024,
        # Caller-controlled: extract_fields calls this twice, first cheaply
        # (max_retries=0, a single attempt) then with the remaining budget
        # only if that attempt's raw text wasn't salvageable — see the
        # comment there for why. The fast Turbo model swapped in for latency
        # (see config.py) is live-measured at only ~65-70% raw success per
        # attempt at emitting a real tool call in instructor's TOOLS mode (vs.
        # ~100% but 30-75s/call for the slower model previously here).
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result


class GraphNodeCreate(BaseModel):
    id: str = Field(
        description=(
            "Short unique snake_case slug for this node, e.g. 'style_modern' or "
            "'accent_navy'. Never use this for the project or a room/budget — "
            "those already exist as anchors; reference their given id instead "
            "of creating a new node for them."
        )
    )
    label: str = Field(description="Human-readable label, e.g. 'Modern style' or 'Navy accent wall'.")
    type: Literal["room", "preference", "constraint", "attribute", "entity"] = "attribute"


class GraphEdgeCreate(BaseModel):
    source: str = Field(description="id of the source node — an anchor id, a recent node's id, or a new node's id from this response")
    target: str = Field(description="id of the target node — an anchor id, a recent node's id, or a new node's id from this response")
    relation: GraphRelation = Field(description="the relationship type connecting source to target")


class GraphNodeRevise(BaseModel):
    """A correction to a fact already captured by a CANDIDATE node — see the
    'no "instead" phrasing required' worked example in app.prompts.GRAPH_SYSTEM_PROMPT.
    app/graph.py marks the target node superseded, creates a fresh node with
    new_label, and adds the 'revises' edge automatically — the model never
    emits that edge itself."""

    target_node_id: str = Field(
        description=(
            "id of the CANDIDATE node this message corrects — MUST be copied "
            "exactly from a candidate id you were shown, never invented."
        )
    )
    new_label: str = Field(description="the corrected label, e.g. 'Matte lacquer finish'")


class GraphExtraction(BaseModel):
    new_nodes: list[GraphNodeCreate] = Field(default_factory=list)
    new_edges: list[GraphEdgeCreate] = Field(default_factory=list)
    # Corrections to existing CANDIDATE nodes (see app.prompts.GRAPH_SYSTEM_PROMPT) —
    # fires on ANY correction, not just explicit "X instead of Y" phrasing.
    revised_nodes: list[GraphNodeRevise] = Field(default_factory=list)
    # Facts the user withdrew with no replacement (e.g. "never mind the
    # accent wall") — ids MUST come from the candidate list, same rule as
    # revised_nodes.
    retracted_node_ids: list[str] = Field(default_factory=list)


# The only types GraphNodeCreate actually allows for a new node — anything
# else (in practice: 'budget' or 'project', the two ANCHOR-only types the
# model sees in its own anchor list and sometimes reaches for) gets coerced
# to the safe default instead of failing validation outright.
_VALID_NEW_NODE_TYPES = {"room", "preference", "constraint", "attribute", "entity"}


def _coerce_and_validate_graph_extraction(payload: dict) -> Optional[GraphExtraction]:
    """Fixes the one schema mismatch actually observed in production traffic
    (see the comment on _salvage_extracted_graph_links) by coercing an
    anchor-only type to 'attribute' before validating. Any other validation
    problem still fails here and returns None — this is a targeted fix for a
    known failure shape, not a blanket bypass of the schema."""
    for node in payload.get("new_nodes") or []:
        if isinstance(node, dict) and node.get("type") not in _VALID_NEW_NODE_TYPES:
            node["type"] = "attribute"
    try:
        return GraphExtraction.model_validate(payload)
    except ValueError:
        return None


def _salvage_extracted_graph_links(
    exc: InstructorRetryException, capture: dict | None = None
) -> Optional[GraphExtraction]:
    """Recovers a usable result from a failed extract_graph_links call instead
    of discarding the whole turn's graph update. Two distinct failure modes,
    both live-observed via Langfuse:

    (1) the model's answer arrives correctly filled in but as its own native
    '<function=...>{...}</function>' text syntax instead of a real tool call,
    which instructor's TOOLS mode rejects outright and can't parse — same
    class of failure _salvage_extracted_fields already handles for the
    fields path. Recovered from message.content.

    (2) a real, well-formed tool call that fails schema validation on
    new_nodes[].type — the model reaches for 'budget' or 'project' (types it
    saw on the ANCHOR nodes it was shown) even though those aren't offered as
    choices for a new node. Retrying with the identical request doesn't help
    here: instructor just resends the same validation error and the model
    repeats the identical mistake (one production trace: 6/6 retries, the
    same 'budget' error every time, 81s spent for a result that got thrown
    away anyway). Recovered from message.tool_calls, which the text-only
    check above never looks at since a real tool call leaves .content empty.
    """
    for attempt in exc.failed_attempts or []:
        completion = getattr(attempt, "completion", None)
        if not completion or not completion.choices:
            continue
        message = completion.choices[0].message

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            arguments = tool_calls[0].function.arguments
            duplicates = _duplicate_json_keys(arguments)
            if duplicates:
                logger.warning(
                    "extract_graph_links salvage: duplicate key(s) %s in tool call arguments — "
                    "only the LAST value survives JSON parsing. raw=%s", duplicates, arguments[:500],
                )
            try:
                payload = json.loads(arguments)
            except (ValueError, TypeError, IndexError, AttributeError):
                payload = None
            if payload is not None:
                result = _coerce_and_validate_graph_extraction(payload)
                if result is not None:
                    if capture is not None:
                        capture["raw_output"] = arguments
                    return result

        content = message.content
        if content:
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = content[start : end + 1]
                duplicates = _duplicate_json_keys(raw)
                if duplicates:
                    logger.warning(
                        "extract_graph_links salvage: duplicate key(s) %s in model output — "
                        "only the LAST value survives JSON parsing. raw=%s", duplicates, raw[:500],
                    )
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = None
                if payload is not None:
                    result = _coerce_and_validate_graph_extraction(payload)
                    if result is not None:
                        if capture is not None:
                            capture["raw_output"] = raw
                        return result
    return None


async def extract_graph_links(
    message: str,
    anchors: list[dict],
    recent_nodes: list[dict],
    recent_edges: list[dict],
    *,
    capture: dict | None = None,
) -> tuple[GraphExtraction, str]:
    """Returns (extraction, status) where status is "clean" (no retry needed)
    or "salvaged" (recovered from a failed attempt's raw text) — app/graph.py
    records this in the turn's trace. Raises InstructorRetryException if both
    the cheap first attempt and the full retry budget are exhausted with
    nothing salvageable; the caller (update_context_graph_node) treats that as
    "skip this turn's LLM extraction," same as extract_fields_node already
    does for the fields path.

    anchors are bounded by room/budget count; recent_nodes (despite the
    parameter name) is now a small top-k relevance search over the ENTIRE
    active graph rather than a recency slice (see app/graph.py::get_candidates)
    — this is what lets a correction to something said many turns ago still
    resolve to its node, while recent_edges stays a small recency window for
    continuity only. Either way the prompt stays flat instead of growing with
    session length."""
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            result = await _extract_graph_links_raw(
                message, anchors, recent_nodes, recent_edges, max_retries, capture=capture
            )
            return result, "clean"
        except InstructorRetryException as exc:
            salvaged = _salvage_extracted_graph_links(exc, capture=capture)
            if salvaged is not None:
                return salvaged, "salvaged"
            last_exc = exc
    raise last_exc


async def _extract_graph_links_raw(
    message: str,
    anchors: list[dict],
    recent_nodes: list[dict],
    recent_edges: list[dict],
    max_retries: int,
    capture: dict | None = None,
) -> GraphExtraction:
    messages = [
        {"role": "system", "content": prompts.GRAPH_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.graph_links_user(anchors, recent_nodes, recent_edges, message)},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_extraction,
        response_model=GraphExtraction,
        max_tokens=1024,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result


async def generate_question(
    field_names: list[str],
    context: dict,
    *,
    is_retry: bool = False,
    capture: dict | None = None,
) -> str:
    """field_names is one or more field labels to gather in a single combined
    question — see app.prompts.question_system."""
    messages = [
        {"role": "system", "content": prompts.question_system(field_names, is_retry)},
        {"role": "user", "content": prompts.question_user(context)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        # GLM-4.7-Flash is a reasoning model that spends tokens on internal
        # chain-of-thought (returned separately as reasoning_content) before
        # emitting the final answer; too low a budget truncates before any
        # visible content is produced (finish_reason="length", content="").
        max_tokens=1024,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


async def generate_wrapup_message(context: dict, *, capture: dict | None = None) -> str:
    """The second of the two question 'types': unlike generate_question (a
    genuine ask that needs an answer), this is a closing statement produced
    once, when save_project_node has just finished the intake — replaces the
    old circle-back re-ask of a skipped field with a plain acknowledgment
    instead, since that field is being silently filled via
    infer_missing_field rather than asked about again. Reuses
    model_question_gen (the fast Turbo model, not model_answer) since this is
    a short, low-stakes line, not a real answer — keeping it off the slower
    model matters for the same latency reasons documented on model_question_gen
    in config.py."""
    messages = [
        {"role": "system", "content": prompts.WRAPUP_PERSONA},
        {"role": "user", "content": prompts.wrapup_user(context)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


async def merge_response(parts: list[str], *, capture: dict | None = None) -> str:
    """Used by app.execution.merge_task_results (Phase 11) to compose several
    deterministic/generated pieces from one multi-task turn into one reply.
    Reuses model_question_gen (fast Turbo model), same latency reasoning as
    generate_wrapup_message — this is short, low-stakes composition, not a
    real answer needing the slower model."""
    messages = [
        {"role": "system", "content": prompts.MERGE_PERSONA},
        {"role": "user", "content": prompts.merge_user(parts)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""






async def infer_missing_field(field_name: str, context: dict, *, capture: dict | None = None) -> str:
    """Called via app.inference.infer_field — produces a single reasonable
    value grounded in whatever's already known (room type, style, budget,
    typical ranges), so a field can be filled without a user answer. Not
    wired into the live decline path today (that's now a whole-room skip,
    see app.graph.classify_intent_node) — infer_field/infer_missing_field
    stay independently available and tested (tests/test_fact_inference_calculation.py)
    for other callers. The caller stores this with changed_by="inferred",
    not "user_message"."""
    messages = [
        {"role": "system", "content": prompts.INFER_MISSING_FIELD_SYSTEM},
        {"role": "user", "content": prompts.infer_missing_field_user(context, field_name)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_extraction,
        messages=messages,
        temperature=0.3,
        max_tokens=64,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return (resp.choices[0].message.content or "").strip()


async def generate_answer(
    message: str, context: dict, history: str = "", retrieved: str = "", *, capture: dict | None = None
) -> AsyncIterator[str]:
    messages = [
        {"role": "system", "content": prompts.GENERATE_ANSWER_SYSTEM},
        {"role": "user", "content": prompts.generate_answer_user(context, retrieved, history, message)},
    ]
    if capture is not None:
        capture["messages"] = messages
    stream = await client.chat.completions.create(
        model=settings.model_answer,
        messages=messages,
        stream=True,
        extra_body=_REASONING_KWARGS,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


async def embed(texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model=settings.model_embedding, input=texts)
    return [d.embedding for d in resp.data]


async def vision(image_url: str, prompt: str, *, capture: dict | None = None) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    resp = await client.chat.completions.create(
        model=settings.model_vision,
        messages=messages,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""
