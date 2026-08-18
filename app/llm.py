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


def _raw_completion_text(completion) -> str:
    """Best-effort raw text of a chat completion — the tool-call arguments
    JSON when the model made a real tool call (TOOLS mode structured output),
    else plain message content. Shared by every structured-output call site
    in this module (classify_operations, resolve_operations,
    resolve_context_changes, ...) for its
    `capture["raw_output"]` trace value."""
    try:
        message = completion.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls[0].function.arguments
    return message.content or ""


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


OperationIntent = Literal[
    "CONTEXT_UPDATE", "CONTEXT_DELETE", "CONTEXT_RETRIEVAL", "DATABASE_RETRIEVAL", "DIRECT_ANSWER"
]


class Operation(BaseModel):
    # One short sentence from the model justifying its connection/confusion
    # calls for this operation (app.prompts.CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE's
    # output.reasoning) — stripped before use downstream, kept only for
    # trace/debugging.
    reasoning: str = ""
    text: str = Field(description="the original meaningful operation text, preserving the user's wording")
    intent: OperationIntent
    # Grounds this operation to a spot in the project tree, per the
    # project-state text classify_operations_user() feeds the model — a
    # root-relative canonical path ("Rooms.a1b2c3d4.Materials.countertop"),
    # a not-yet-existing entity name ("Living Room"), or None when the model
    # can't confidently place it. app.graph's clarifying-question branch
    # asks the user directly for every operation where this is None.
    connection: Optional[str] = None
    # True when the user stated a genuine unresolved either/or CONTENT
    # decision (see CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE's CONFUSION section)
    # — independent of `connection`: an operation can be ungrounded,
    # content-undecided, both, or neither. app.graph's clarifying-question
    # branch holds the turn for this too, alongside connection is None.
    confusion: bool = False
    # Short description of the specific fork, required (non-null) when
    # confusion is True — feeds the Resolution Agent's content-question
    # generation. Always None when confusion is False.
    confusion_note: Optional[str] = None
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
        # gpt-oss-120b is a reasoning model — its hidden reasoning tokens
        # (see _REASONING_KWARGS) are drawn from this same max_tokens budget
        # before any visible output is produced. 1024 was enough for a
        # short, single-operation message but silently truncated (provider
        # "length" finish reason -> failed JSON validation -> instructor
        # retry exhaustion) once a longer message splits into many
        # operations, each with its own reasoning/text/intent/connection
        # fields. Sized generously since gpt-oss-120b's context window has
        # ample headroom for this.
        max_tokens=4096,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.operations


class Option(BaseModel):
    # None for an inferred (not-yet-existing) room, or for any "content"
    # resolution option (see app.prompts.RESOLUTION_AGENT_SYSTEM_TEMPLATE) —
    # a real existing room always carries its exact room id. Left None on a
    # bundled multi-room option too (room_ids set instead, see below).
    id: Optional[str] = None
    label: str
    # Set ONLY on a bundled "room" option that applies the same operation to
    # SEVERAL existing rooms at once (app.prompts.RESOLUTION_AGENT_SYSTEM_
    # TEMPLATE's multi-room rule) — the exact room ids of every bundled
    # room, copied verbatim from the tree, never an inferred/new room. None
    # for an ordinary single-room option, and always None on a "content"
    # option. app.graph.classify_intent_node's resume branch fans a chosen
    # bundle out into one write task per room id.
    room_ids: Optional[list[str]] = None


class ResolutionItem(BaseModel):
    """One open question for one operation, from resolve_operations() — see
    app.prompts.RESOLUTION_AGENT_SYSTEM_TEMPLATE. `id` mirrors the input
    operation's own id and is how the caller correlates results back — NOT
    positional matching, since an operation needing both a room and a
    content question returns TWO items sharing the same `id`. Always carries
    a question and at least one option, even when a room is confidently
    resolved ("confirm this one") — there is no "resolvable without asking"
    verdict in this prompt, which is what keeps app.graph.classify_intent_node's
    always-block posture (see the classifier-connection-always-block memory)
    a property of the prompt itself rather than something the caller has to
    enforce by ignoring a confidence flag."""

    id: str
    resolution_type: Literal["room", "content"]
    text: str
    intent: str
    question: str
    options: list[Option] = Field(default_factory=list)


class ResolutionBatch(BaseModel):
    resolutions: list[ResolutionItem] = Field(default_factory=list)


async def _salvage_resolutions(
    exc: InstructorRetryException, capture: dict | None = None
) -> Optional[list[ResolutionItem]]:
    """Same recovery strategy as _salvage_operations — scans every failed
    attempt for a real tool call or a JSON array embedded in plain text
    content, so a parse hiccup doesn't silently drop the whole batch's worth
    of questions."""
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
            result = [ResolutionItem.model_validate(item) for item in items]
        except (ValueError, TypeError, KeyError):
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result
    return None


async def resolve_operations(
    tree_text: str,
    operations: list[dict],
    *,
    capture: dict | None = None,
) -> list[ResolutionItem]:
    """Every operation this turn that classify_operations left with
    connection is None, confusion is True, or both, resolved in ONE batched
    call — replaces the old generate_clarification_question (one call per
    op) and resolve_room_connections (room-only). `operations` is
    `[{"id", "text", "intent", "connection", "confusion", "confusion_note"}, ...]`.
    Results are grouped by `id` by the caller (app.graph._generate_operation_questions),
    NOT positionally — an operation needing both a room and content question
    comes back as two ResolutionItems sharing one id."""
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            return await _resolve_operations_raw(tree_text, operations, max_retries, capture=capture)
        except InstructorRetryException as exc:
            salvaged = await _salvage_resolutions(exc, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _resolve_operations_raw(
    tree_text: str,
    operations: list[dict],
    max_retries: int,
    capture: dict | None = None,
) -> list[ResolutionItem]:
    operations_json = json.dumps(operations, indent=2)
    messages = [
        {"role": "system", "content": prompts.resolution_agent_system(tree_text, operations_json)},
        {"role": "user", "content": prompts.resolution_agent_user()},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_intent_classifier,
        response_model=ResolutionBatch,
        # See the matching comment in _classify_operations_raw — this batch
        # can carry a "room" AND "content" ResolutionItem per open
        # operation, so it needs the same headroom for a message that splits
        # into many operations.
        max_tokens=4096,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.resolutions


# ---------------------------------------------------------------------------
# resolve_context_confusion — the resume-turn answer resolver
# (app.prompts.RESOLVE_CONTEXT_CONFUSSION). Takes every operation still
# holding the turn after a resume (a free-text answer, or any operation that
# had a confusion:true question pending — see app.graph.classify_intent_node's
# resume branch for why those never take the cheap deterministic-merge path)
# plus each of its pending room/content question(s) and the user's answer,
# and returns ONE finalized operation per input — connection guaranteed
# non-null, confusion guaranteed false. No "still open" outcome exists.
# ---------------------------------------------------------------------------


class PendingResolution(BaseModel):
    resolution_type: Literal["room", "content"]
    question: str
    # Room options carry a precomputed "connection_path" (app.graph
    # attaches it from the live tree before this call — see
    # RESOLVE_CONTEXT_CONFUSSION's own note not to reconstruct it); content
    # options don't. Left as plain dicts (not Option) since the shape
    # differs from resolve_operations' own Option and is call-site-specific.
    options: list[dict] = Field(default_factory=list)
    user_answer: str


class OperationToResolve(BaseModel):
    id: str
    original_operation: dict
    pending_resolutions: list[PendingResolution] = Field(default_factory=list)


class ResolvedOperation(BaseModel):
    id: str
    reasoning: str = ""
    text: str
    intent: str
    connection: str
    confusion: bool = False
    confusion_note: Optional[str] = None


class ResolvedOperationBatch(BaseModel):
    resolved_operations: list[ResolvedOperation] = Field(default_factory=list)


async def _salvage_resolved_operations(
    exc: InstructorRetryException, capture: dict | None = None
) -> Optional[list[ResolvedOperation]]:
    """Same recovery strategy as _salvage_resolutions/_salvage_operations."""
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
            result = ResolvedOperationBatch.model_validate(payload)
        except (ValueError, TypeError):
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result.resolved_operations
    return None


async def resolve_context_confusion(
    tree_text: str,
    operations: list[dict],
    *,
    capture: dict | None = None,
) -> list[ResolvedOperation]:
    """`operations` is `[OperationToResolve.model_dump(), ...]`. Every
    returned ResolvedOperation is fully resolved — connection non-null,
    confusion false — so app.graph's resume branch never needs a further
    branch after merging these back into `tasks` by id."""
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            return await _resolve_context_confusion_raw(tree_text, operations, max_retries, capture=capture)
        except InstructorRetryException as exc:
            salvaged = await _salvage_resolved_operations(exc, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _resolve_context_confusion_raw(
    tree_text: str,
    operations: list[dict],
    max_retries: int,
    capture: dict | None = None,
) -> list[ResolvedOperation]:
    operations_json = json.dumps(operations, indent=2)
    messages = [
        {"role": "system", "content": prompts.resolve_context_confusion_system(tree_text, operations_json)},
        {"role": "user", "content": prompts.resolve_context_confusion_user()},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_intent_classifier,
        response_model=ResolvedOperationBatch,
        # See the matching comment in _classify_operations_raw.
        max_tokens=4096,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.resolved_operations


# ---------------------------------------------------------------------------
# resolve_context_changes — the pipeline's single combined call covering
# every CONTEXT_UPDATE and CONTEXT_DELETE operation in a turn (see
# app.pipeline._run_first_action). Replaces the old per-message extract_fields
# + extract_graph_links pair AND app.canonical_mapper.resolve_deletion_target's
# embedding-similarity delete matching — deletion targets are picked directly
# from the live tree text here instead.
#
# Deliberately narrower than the old ExtractedFields: classify_operations has
# already split a multi-room message into separate, individually-grounded
# operations (each with its own `connection`), so this schema never needs to
# juggle a SECOND room's figure in the same result the way the old
# additionalRoomBudgets/mentionedAdditionalRooms had to — a second room's
# figure is just another operation in the same batch, with its own
# connection. Room identity itself is never re-derived here either: the
# caller resolves each operation's room from its own `connection` via
# app.canonical_mapper.split_connection, same as every other write path.
# ---------------------------------------------------------------------------


class MaterialChange(BaseModel):
    item: str
    material: str
    specification: Optional[str] = None


class ContextChangeFields(BaseModel):
    """Structured field values for ONE CONTEXT_UPDATE operation, scoped to
    that operation's own room (resolved separately from `connection`, not
    here)."""

    projectType: Optional[str] = Field(
        None, description="new construction, renovation, or a partial refresh/update — only when this operation is Project-scoped"
    )
    overallBudget: Optional[str] = Field(
        None, description="the TOTAL/whole-project budget — only when this operation is Project-scoped, never a single room's own figure"
    )
    timeline: Optional[str] = Field(None, description="only when this operation is Project-scoped")
    budgetOrRequirement: Optional[str] = Field(None, description="budget or need for THIS operation's own room specifically")
    style: Optional[str] = None
    squareFootage: Optional[float] = None
    existingFurniture: Optional[str] = None
    materials: Optional[list[MaterialChange]] = Field(None, description="items newly mentioned in THIS operation only")


class FreeformEntityChange(BaseModel):
    raw_entity: str = Field(description="the entity as the user described it, e.g. 'walnut TV cabinet'")
    node_type_hint: Optional[Literal["Materials", "Furniture", "Attributes", "Constraints", "ClientPreferences"]] = Field(
        None, description="best-guess category for a NEW entity (existing_path null) — leave null if genuinely unclear, it will be inferred downstream"
    )
    # existing_path/field/value: the entity-edit path — mirrors
    # deletion_targets' own "exact path copied verbatim, never invented"
    # contract (see app.pipeline._apply_update, which writes `field` on the
    # entity at `existing_path` directly rather than routing through
    # canonical_mapper.map_to_canonical). Left null (the default) for a
    # genuinely new mention, which still goes through map_to_canonical's
    # algorithmic alias/embedding dedup exactly as before.
    existing_path: Optional[str] = Field(
        None,
        description=(
            "exact canonical path copied verbatim from CURRENT DATA TREE if this mention refers to an entity that "
            "ALREADY EXISTS there (the user is editing it, not introducing something new) — null for a genuinely "
            "new mention. Never invent a path; if you're not confident the entity is the same one shown in the "
            "tree, leave this null instead of guessing."
        ),
    )
    field: Optional[Literal["Label", "Material", "Specification", "Quantity", "Notes"]] = Field(
        None,
        description="set together with `value` ONLY when existing_path is set and the user is changing one specific leaf of that entity — both null otherwise",
    )
    value: Optional[str] = Field(None, description="the new value for `field` — set together with `field`, both null otherwise")
    # Composite-entity support (Parts/Properties/Quantity). When the
    # operation's `connection` points at an existing freeform instance (e.g.
    # "Rooms.<id>.Furniture.beds") AND the text attaches something TO that
    # instance ("add a bedcover to the bed"), emit the attached thing as a
    # `parts` entry — app.pipeline._apply_update routes each part through
    # canonical_mapper.map_part_to_canonical, nesting it under the parent
    # instance's Parts container instead of as a sibling under the room.
    # `quantity` is the user-stated count for THIS entity itself ("two
    # pillows" -> "2"); `properties` are named key/value attributes (color,
    # finish, fabric) not covered by the typed leaf fields. All three are
    # additive on top of the existing edit/new-mention shape — a plain
    # non-composite mention leaves them empty/null exactly as before.
    quantity: Optional[str] = Field(
        None, description="a count the user stated for THIS entity, as a string (e.g. 'two pillows' -> '2') — null when no count was given"
    )
    properties: list["PropertyChange"] = Field(
        default_factory=list,
        description="named key/value attributes for this entity (color, finish, fabric, etc.) not covered by the typed leaf fields — empty when none stated",
    )
    parts: list["PartChange"] = Field(
        default_factory=list,
        description="physical sub-components attached TO this entity (a sofa's cover, a bed's bedcover, pillows on top of it) — one level only. Empty when the mention is not composite.",
    )


class PropertyChange(BaseModel):
    """A named key/value attribute attached to a freeform entity or one of
    its parts — color, finish, fabric, or any ad-hoc property. Generic by
    design (no per-attribute typed field): `name` is the property's own
    name ("color"), `value` is its value ("red"). app.pipeline writes each
    as a Properties.<slug> instance (Label=name, Value=value) nested under
    the entity/part via canonical_mapper.map_property_to_canonical."""
    name: str = Field(description="the property's name, e.g. 'color', 'finish', 'fabric'")
    value: str = Field(description="the property's value, e.g. 'red', 'matte', 'velvet'")


class PartChange(BaseModel):
    """A physical sub-component attached to a parent freeform entity — a
    sofa's cover, a bed's bedcover, pillows on top of something. ONE LEVEL
    ONLY: this model deliberately has NO `parts` field, so a sub-component
    of a part (e.g. 'pillow with a zipper') is folded into the part's own
    `properties`/`quantity`/`notes` rather than nesting further — enforced
    structurally, not just by prompt. `existing_path` mirrors
    FreeformEntityChange.existing_path for editing an existing part's leaf
    (Label/Material/Quantity/Notes); null for a genuinely new part."""
    raw_entity: str = Field(description="the part as the user described it, e.g. 'bedcover', 'velvet cover'")
    material: Optional[str] = Field(None, description="the material/finish stated for this part, e.g. 'velvet' — null when none stated")
    quantity: Optional[str] = Field(
        None, description="a count the user stated for this part, as a string ('two pillows' -> '2') — null when no count was given"
    )
    properties: list[PropertyChange] = Field(
        default_factory=list,
        description="named key/value attributes for this part — same shape as FreeformEntityChange.properties",
    )
    existing_path: Optional[str] = Field(
        None,
        description=(
            "exact canonical path copied verbatim from CURRENT DATA TREE if this part ALREADY EXISTS under its "
            "parent and the user is editing one of its leaves — null for a genuinely new part. Never invent a path."
        ),
    )
    field: Optional[Literal["Label", "Material", "Quantity", "Notes"]] = Field(
        None,
        description="set together with `value` ONLY when existing_path is set and the user is editing one specific leaf of that part — both null otherwise",
    )
    value: Optional[str] = Field(None, description="the new value for `field` — set together with `field`, both null otherwise")


FreeformEntityChange.model_rebuild()
PartChange.model_rebuild()


class ContextChangeResult(BaseModel):
    text: str = Field(description="copied from the input operation's own text")
    intent: Literal["CONTEXT_UPDATE", "CONTEXT_DELETE"] = Field(description="copied from the input operation's own intent")
    fields: ContextChangeFields = Field(default_factory=ContextChangeFields, description="CONTEXT_UPDATE only — leave every field null for CONTEXT_DELETE")
    freeform_entities: list[FreeformEntityChange] = Field(
        default_factory=list, description="CONTEXT_UPDATE only — materials/furniture/attributes/constraints/preferences not covered by `fields`"
    )
    deletion_targets: list[str] = Field(
        default_factory=list,
        description=(
            "CONTEXT_DELETE only — exact canonical path(s) copied verbatim from CURRENT DATA TREE that this "
            "operation retracts. Empty if no confident target exists in the tree. Never invent a path."
        ),
    )


class ContextChangeBatch(BaseModel):
    results: list[ContextChangeResult] = Field(default_factory=list)


class ContextChangeValidationError(BaseModel):
    """One semantic-validation failure from _validate_context_changes — see
    that function's docstring. `index` is the position within whichever list
    `kind` names (deletion_targets / freeform_entities / parts) on
    results[result_index]. For kind="part", `entity_index` names which
    freeform_entities entry the part belongs to (parts are nested, so a
    single int isn't enough to locate them); null for the other kinds."""

    result_index: int
    kind: Literal["deletion_target", "freeform_entity", "part"]
    index: int
    detail: str
    entity_index: Optional[int] = None


async def _salvage_context_changes(
    exc: InstructorRetryException, capture: dict | None = None
) -> Optional[list[ContextChangeResult]]:
    """Same recovery strategy as _salvage_room_resolutions — scans every
    failed attempt for a real tool call or a JSON array/object embedded in
    plain text content, so a parse hiccup doesn't silently drop the whole
    batch's worth of context changes."""
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
            start = min((i for i in (content.find("["), content.find("{")) if i != -1), default=-1)
            end = max(content.rfind("]"), content.rfind("}"))
            if start != -1 and end != -1 and end > start:
                raw = content[start : end + 1]

        if not raw:
            continue
        try:
            payload = json.loads(raw)
            items = payload["results"] if isinstance(payload, dict) else payload
            result = [ContextChangeResult.model_validate(item) for item in items]
        except (ValueError, TypeError, KeyError):
            continue
        if capture is not None:
            capture["raw_output"] = raw
        return result
    return None


def _validate_context_changes(
    results: list[ContextChangeResult],
    existing_paths: set[str],
    node_types_by_path: dict[str, str],
) -> list[ContextChangeValidationError]:
    """Semantic validation beyond Pydantic's structural checks — the LLM
    only ever names two kinds of path in this schema (deletion_targets,
    freeform_entities[].existing_path), so this only checks those actually
    exist in the CURRENT DATA TREE snapshot the caller fetched, plus that an
    entity edit's `field` is legal for that entity's own node_type
    (app.canonical_mapper.legal_fields). node_type/intent legality is
    already a closed Pydantic Literal — nothing to re-check there. Pure:
    reads only what's passed in, same as every other function in this
    module never touching graph_store directly — the caller
    (app.pipeline._run_first_action) is the one with DB access."""
    from app import canonical_mapper  # lazy: canonical_mapper imports this module for embed(), avoid a load-time cycle

    issues: list[ContextChangeValidationError] = []
    for i, result in enumerate(results):
        for j, path in enumerate(result.deletion_targets):
            if path not in existing_paths:
                issues.append(ContextChangeValidationError(
                    result_index=i, kind="deletion_target", index=j,
                    detail=f'"{path}" does not exist in CURRENT DATA TREE',
                ))
        for j, mention in enumerate(result.freeform_entities):
            if mention.existing_path is None:
                if mention.field is not None:
                    issues.append(ContextChangeValidationError(
                        result_index=i, kind="freeform_entity", index=j,
                        detail=f'field="{mention.field}" set without existing_path',
                    ))
                continue
            if mention.existing_path not in existing_paths:
                issues.append(ContextChangeValidationError(
                    result_index=i, kind="freeform_entity", index=j,
                    detail=f'existing_path "{mention.existing_path}" does not exist in CURRENT DATA TREE',
                ))
            elif mention.field is not None:
                node_type = node_types_by_path.get(mention.existing_path)
                allowed = canonical_mapper.legal_fields(node_type) if node_type else []
                if mention.field not in allowed:
                    issues.append(ContextChangeValidationError(
                        result_index=i, kind="freeform_entity", index=j,
                        detail=f'field="{mention.field}" is not legal for "{mention.existing_path}" (node_type={node_type!r}); allowed: {allowed}',
                    ))
            # Parts: a part's existing_path must be a real path in the tree
            # AND its field (if set) legal for the Parts node_type. Same
            # contract as the entity-level check above, just one level down —
            # the LLM only ever names existing_path on a PartChange when it
            # is editing an existing part's leaf, never to claim a parent.
            # kind="part" + entity_index=j so _drop_invalid_claims removes
            # just the bad part, not the whole parent mention.
            for k, part in enumerate(mention.parts):
                if part.existing_path is None:
                    if part.field is not None:
                        issues.append(ContextChangeValidationError(
                            result_index=i, kind="part", index=k, entity_index=j,
                            detail=f'parts[{k}].field="{part.field}" set without existing_path',
                        ))
                    continue
                if part.existing_path not in existing_paths:
                    issues.append(ContextChangeValidationError(
                        result_index=i, kind="part", index=k, entity_index=j,
                        detail=f'parts[{k}].existing_path "{part.existing_path}" does not exist in CURRENT DATA TREE',
                    ))
                elif part.field is not None:
                    node_type = node_types_by_path.get(part.existing_path)
                    allowed = canonical_mapper.legal_fields(node_type) if node_type else []
                    if part.field not in allowed:
                        issues.append(ContextChangeValidationError(
                            result_index=i, kind="part", index=k, entity_index=j,
                            detail=f'parts[{k}].field="{part.field}" is not legal for "{part.existing_path}" (node_type={node_type!r}); allowed: {allowed}',
                        ))
    return issues


def _drop_invalid_claims(results: list[ContextChangeResult], issues: list[ContextChangeValidationError]) -> None:
    """Mutates `results` in place, removing exactly the claims `issues`
    flagged — a bad deletion_targets entry is removed from that list; a
    freeform_entities mention with a bad existing_path/field is dropped
    entirely (never silently re-purposed into a create — the user's text was
    about editing something that already exists, so falling through to
    map_to_canonical would create something they never asked for)."""
    by_result: dict[int, list[ContextChangeValidationError]] = {}
    for issue in issues:
        by_result.setdefault(issue.result_index, []).append(issue)

    for i, result_issues in by_result.items():
        result = results[i]
        bad_deletion_indices = {issue.index for issue in result_issues if issue.kind == "deletion_target"}
        if bad_deletion_indices:
            result.deletion_targets = [p for j, p in enumerate(result.deletion_targets) if j not in bad_deletion_indices]
        bad_entity_indices = {issue.index for issue in result_issues if issue.kind == "freeform_entity"}
        if bad_entity_indices:
            result.freeform_entities = [m for j, m in enumerate(result.freeform_entities) if j not in bad_entity_indices]
        # Parts are nested under a freeform_entity, so drop the bad part from
        # its parent's `parts` list (not the whole parent mention) — group by
        # entity_index so one pass per parent mutates its list in place.
        part_issues_by_entity: dict[int, set[int]] = {}
        for issue in result_issues:
            if issue.kind == "part" and issue.entity_index is not None:
                part_issues_by_entity.setdefault(issue.entity_index, set()).add(issue.index)
        for entity_index, bad_part_indices in part_issues_by_entity.items():
            if entity_index < len(result.freeform_entities):
                mention = result.freeform_entities[entity_index]
                mention.parts = [p for k, p in enumerate(mention.parts) if k not in bad_part_indices]


def _format_validation_issues(operations: list[dict], issues: list[ContextChangeValidationError]) -> str:
    lines = []
    for issue in issues:
        op_text = operations[issue.result_index].get("text", "") if issue.result_index < len(operations) else "?"
        lines.append(f'- operation {issue.result_index} ("{op_text}"), {issue.kind} #{issue.index}: {issue.detail}')
    return "\n".join(lines)


async def resolve_context_changes(
    tree_text: str,
    operations: list[dict],
    existing_paths: set[str],
    node_types_by_path: dict[str, str],
    *,
    capture: dict | None = None,
) -> tuple[list[ContextChangeResult], list[ContextChangeValidationError]]:
    """Every CONTEXT_UPDATE/CONTEXT_DELETE operation in this turn, resolved
    in ONE batched call — `operations` is
    `[{"text", "intent", "connection"}, ...]`, in the exact order results
    come back in (positional matching, no id round-trip needed, same
    convention resolve_room_connections already uses).

    Two validation layers, matching the function-calling-implementation
    guide's design: `_resolve_context_changes_structural` below is Pydantic/
    instructor's structural contract (unchanged retry+salvage behavior);
    `_validate_context_changes` is the semantic layer on top of that (does a
    claimed path actually exist? is a claimed field legal for it?). On a
    semantic failure, ONE retry is made with the exact validation error fed
    back into the prompt; anything still invalid after that retry has its
    specific bad claim(s) dropped (app.pipeline never sees a path it should
    trust that wasn't actually verified) — the caller surfaces the returned
    issues however it sees fit (app.pipeline._run_first_action turns each
    into a `changes` entry with action="failed"). Never raises for a
    semantic failure — only a structural failure that survives both the
    instructor retry budget AND salvage still propagates as
    InstructorRetryException, same as before this validation layer existed."""
    results = await _resolve_context_changes_structural(tree_text, operations, capture=capture)

    issues = _validate_context_changes(results, existing_paths, node_types_by_path)
    if not issues:
        return results, []

    logger.warning(
        "resolve_context_changes: %d validation issue(s) on first attempt, retrying once — %s",
        len(issues), [i.detail for i in issues],
    )
    error_summary = _format_validation_issues(operations, issues)
    retry_results = await _resolve_context_changes_structural(tree_text, operations, capture=capture, validation_errors=error_summary)

    retry_issues = _validate_context_changes(retry_results, existing_paths, node_types_by_path)
    if retry_issues:
        logger.warning(
            "resolve_context_changes: %d validation issue(s) still present after retry, dropping those claims — %s",
            len(retry_issues), [i.detail for i in retry_issues],
        )
        _drop_invalid_claims(retry_results, retry_issues)
    return retry_results, retry_issues


async def _resolve_context_changes_structural(
    tree_text: str,
    operations: list[dict],
    *,
    validation_errors: Optional[str] = None,
    capture: dict | None = None,
) -> list[ContextChangeResult]:
    """The structural (Pydantic/instructor) retry+salvage loop — unchanged
    behavior from before the semantic validation layer existed, just
    factored out so resolve_context_changes can call it twice (once clean,
    once with a validation_errors block appended for the semantic-retry
    pass)."""
    last_exc: Optional[InstructorRetryException] = None
    for max_retries in (0, 4):
        try:
            return await _resolve_context_changes_raw(tree_text, operations, max_retries, validation_errors=validation_errors, capture=capture)
        except InstructorRetryException as exc:
            salvaged = await _salvage_context_changes(exc, capture=capture)
            if salvaged is not None:
                return salvaged
            last_exc = exc
    raise last_exc


async def _resolve_context_changes_raw(
    tree_text: str,
    operations: list[dict],
    max_retries: int,
    *,
    validation_errors: Optional[str] = None,
    capture: dict | None = None,
) -> list[ContextChangeResult]:
    operations_json = json.dumps(operations, indent=2)
    messages = [
        {"role": "system", "content": prompts.resolve_context_changes_system(tree_text, operations_json)},
        {"role": "user", "content": prompts.resolve_context_changes_user(validation_errors)},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_extraction,
        response_model=ContextChangeBatch,
        # See the matching comment in _classify_operations_raw — this batch
        # also fans out per operation, so the same headroom applies.
        max_tokens=4096,
        max_retries=max_retries,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result.results


# ---------------------------------------------------------------------------
# generate_search_keywords — DATABASE_RETRIEVAL's new keyword-generation step
# in front of app.rag.query_catalog, which today just embeds the raw message
# verbatim. A short, low-stakes call — reuses model_question_gen (the fast
# Turbo model), same latency posture as generate_wrapup_message/merge_response.
# ---------------------------------------------------------------------------


async def generate_search_keywords(query: str, *, capture: dict | None = None) -> str:
    messages = [
        {"role": "system", "content": prompts.SEARCH_KEYWORDS_SYSTEM},
        {"role": "user", "content": prompts.search_keywords_user(query)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.2,
        max_tokens=64,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# generate_turn_summary — the pipeline's final join step (see
# app.pipeline.run_pipeline): one call that turns whichever pieces this turn
# actually produced (changes made, context retrieved, database results,
# still-open fields) into the turn's actual reply text.
# ---------------------------------------------------------------------------


class TurnSummary(BaseModel):
    database_summary: Optional[str] = Field(None, description="null if no database/catalog search happened this turn")
    context_summary: Optional[str] = Field(None, description="null if no context retrieval happened this turn")
    changes_summary: Optional[str] = Field(None, description="null if nothing was created/updated/deleted this turn")
    next_message: str = Field(description="the next question to ask, or a closing/completion line if nothing is left to ask")
    is_question: bool = Field(description="true if next_message is a question awaiting an answer, false if it's a closing/completion statement")


async def generate_turn_summary(pieces: dict, *, capture: dict | None = None) -> TurnSummary:
    messages = [
        {"role": "system", "content": prompts.TURN_SUMMARY_SYSTEM},
        {"role": "user", "content": prompts.turn_summary_user(pieces)},
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_question_gen,
        response_model=TurnSummary,
        max_tokens=768,
        max_retries=2,
        messages=messages,
        extra_body=_REASONING_KWARGS,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result



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
