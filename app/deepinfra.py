import json
import logging
from collections.abc import AsyncIterator
from typing import Literal, Optional

import instructor
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, Field

from app import observability  # noqa: F401  (constructs the Langfuse singleton before AsyncOpenAI below)
from app.config import settings
from langfuse.openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.deepinfra_api_key, base_url=settings.deepinfra_base_url)
structured_client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)


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

Intent = Literal["context_related", "direct_question", "database_query", "update_context"]
_VALID_INTENTS = {"context_related", "direct_question", "database_query", "update_context"}

_CLASSIFY_SYSTEM = (
    "You are an intent router for an interior design assistant. Classify the user's "
    "latest message into exactly one label — or, rarely, two comma-separated labels "
    "if the message clearly glues together two separate asks (see the last rule "
    "below):\n"
    "- context_related: asks about details already given in this conversation (e.g. "
    "'what's my budget again?')\n"
    "- direct_question: expects an answer — general questions, and ALSO cost/price/"
    "budget/advice questions like 'what would the kitchen cost' or 'what should this "
    "cost for a luxury house' — these want a real answer, not a follow-up question\n"
    "- database_query: asks to find/search/recommend specific products or style "
    "references\n"
    "- update_context: STATES a new or updated project fact/preference (room, budget "
    "number, style, size, furniture, a specific material, timeline) — the user is "
    "telling you something, not asking something\n\n"
    "The single most important rule: if the latest message is ITSELF phrased as a "
    "question (contains 'what', 'how', 'when', 'why', 'should', 'can you', asks the "
    "cost/price/budget of something, etc.), it is NEVER update_context — classify it "
    "by what it's asking about instead. This holds regardless of what the assistant's "
    "previous message asked; a question is always a question, even right after the "
    "assistant asked the user something.\n\n"
    "update_context is only for messages that state a fact, even a brief one with no "
    "question mark — e.g. 'quartz countertops', 'I want oak flooring', '$15k budget', "
    "'600 sqft', including short replies that directly answer a pending question with "
    "an actual fact (not another question).\n\n"
    "A greeting or small talk with no project fact in it (e.g. 'hello', 'hi', 'hey', "
    "'thanks') is direct_question, never update_context — there is nothing to update.\n\n"
    "Two-label rule: only output two labels when the message states a fact AND asks a "
    "separate question in the same breath (e.g. 'what does oak flooring cost, and my "
    "budget is $15k' -> database_query,update_context). A single short answer like "
    "'residential' or 'modern' is always exactly one label (update_context), never two.\n\n"
    "Examples:\n"
    "\"what's my budget again?\" -> context_related\n"
    "\"what should my budget be?\" -> direct_question\n"
    "\"modern\" (pending: room style) -> update_context\n"
    "\"recommend some tile options\" -> database_query\n"
    "\"what would oak flooring cost?\" -> direct_question\n"
    "\"quartz countertops\" -> update_context\n"
    "\"hey\" -> direct_question\n"
    "\"can I get a quote on that walnut console\" -> database_query\n"
    "\"oak flooring for the kitchen, love the modern look\" -> update_context   "
    "(NOT two labels — one continuous fact, no question)\n"
    "\"modern, and keep it under $15k\" -> update_context   "
    "(NOT two labels — both are facts, not a fact+question)\n"
    "\"what does oak flooring cost, and my budget is $15k\" -> database_query,update_context   "
    "(genuine compound: a question AND a separate fact)\n"
    "\"recommend some tile options for my modern kitchen\" -> database_query   "
    "(NOT two labels — the style mention is context for the search, not a separate fact to extract)\n\n"
    "Reply with only the label(s), nothing else."
)


async def classify_intent(
    message: str,
    history: str = "",
    pending_field: str | None = None,
    *,
    capture: dict | None = None,
) -> list[Intent]:
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history}\n\n"
                f"Currently pending question (if any): {pending_field or 'none'}\n\n"
                f"Latest message:\n{message}"
            ),
        },
    ]
    resp = await client.chat.completions.create(
        model=settings.model_intent_classifier,
        messages=messages,
        temperature=0,
        max_tokens=20,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    raw = (resp.choices[0].message.content or "").strip().lower()
    labels: list[Intent] = []
    for piece in raw.split(","):
        label = piece.strip()
        if label in _VALID_INTENTS and label not in labels:
            labels.append(label)  # type: ignore[arg-type]
    return labels or ["direct_question"]


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
        {
            "role": "system",
            "content": (
                "Extract interior design project fields from the user's message. Only "
                "fill fields explicitly stated or clearly implied; leave others null. "
                "Do not invent values — if the message states no new project info (e.g. "
                "it's just a question), return every field null. A hedged or approximate "
                "statement ('maybe around 4 lakh', 'roughly 300 sqft') still counts as "
                "stated — extract it with the hedge wording kept intact rather than "
                "leaving the field null; only leave a field null when the message truly "
                "doesn't address it at all."
            ),
        },
        {
            "role": "user",
            "content": f"Known so far: {known or '(nothing yet)'}\n\nNew message: {message}",
        },
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
    'no "instead" phrasing required' worked example in _GRAPH_SYSTEM_PROMPT.
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
    # Corrections to existing CANDIDATE nodes (see _GRAPH_SYSTEM_PROMPT) —
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


_GRAPH_SYSTEM_PROMPT = (
    "You extend a knowledge graph of an interior design project. You are given "
    "a fixed set of ANCHOR nodes — the project itself, and one per room or "
    "budget the client has mentioned — that already exist and are stable. "
    "Never invent an id for the project or a room/budget, and never create a "
    "new node for one: always reference the exact anchor id you were given. "
    "You are also shown CANDIDATE nodes: freeform facts already captured "
    "(furniture, materials, preferences, constraints, rejected alternatives) "
    "that scored as plausibly related to this new message, gathered by "
    "search across the ENTIRE project history — not just recent turns, so a "
    "candidate may be something said many messages ago if it's relevant to "
    "correcting or extending now. It is NOT the complete history, so don't "
    "assume something is new just because it isn't among the candidates "
    "shown.\n\n"
    "From the user's new message, extract new nodes (facts, preferences, "
    "entities, constraints, or rejected alternatives actually stated or "
    "clearly implied) and edges connecting each one to an anchor id or to a "
    "candidate node id you were given. Do not invent facts not stated.\n\n"
    "If the message instead CORRECTS or refines something a candidate node "
    "already represents, use revise_node — this applies to ANY correction, "
    "not only explicit 'X instead of Y' phrasing: 'let's make it navy', "
    "'actually go with quartz', 'scratch the walnut, do oak', 'change the "
    "budget to 5 lakh' are ALL revisions of an existing candidate if one "
    "matches, not new unconnected facts. target_node_id MUST be copied "
    "exactly from a candidate id you were shown — never invent one, and "
    "never use revise_node against an anchor or a node not in the candidate "
    "list. Prefer revise_node over creating both an "
    "add_node-and-rejected_in_favor_of pair — reserve "
    "rejected_in_favor_of for when the user explicitly wants BOTH the old "
    "and new choice kept visible as a comparison (rare).\n\n"
    "If the user drops something with no replacement ('never mind the "
    "accent wall'), put that candidate's id in retracted_node_ids instead of "
    "creating or revising anything.\n\n"
    "Pick the relation deliberately: 'located_in' when an item belongs to a "
    "room, 'uses_material' when a material is chosen for an item, "
    "'budget_for' when a figure applies to a room or the project, "
    "'applies_to' for a preference/requirement about a room, "
    "'rejected_in_favor_of' when the user explicitly drops one choice for "
    "another, 'requires' for a stated dependency, 'modifies' when one fact "
    "refines another, 'part_of' for plain containment. Every relation reads "
    "child-to-parent: source is the more specific thing, target is what it "
    "belongs to or applies to — so a budget figure's edge always goes "
    "'source: the budget node, target: the room or project it's for', never "
    "the other way around (see the worked example below). Reuse an existing "
    "id (anchor or recent) when the message refers to something already "
    "captured — never duplicate a node for the same concept. Keep new node "
    "ids short snake_case slugs. Preserve concrete numbers and named choices "
    "in the label (e.g. '15 lakh budget', not just 'Budget').\n\n"
    "If the message states a figure for the project overall AND a separate "
    "figure for one specific room, extract BOTH as distinct nodes with "
    "distinct labels and give each its own 'budget_for' edge to its own "
    "target (project vs. that room's anchor) — never merge two different "
    "figures into one node, and never attach a room's own figure to the "
    "project anchor or vice versa.\n\n"
    "A new node's type must be exactly one of: room, preference, constraint, "
    "attribute, entity. Never 'project' or 'budget' — those only exist as "
    "the anchors you were already given, never as something you create. If "
    "the message states a budget figure for a room or the project that "
    "doesn't have an anchor yet, create it as type 'attribute' (not "
    "'budget') and connect it to the closest anchor you do have — the "
    "project anchor if no room anchor exists yet — with relation "
    "'budget_for'.\n\n"
    "When the message says 'both' or 'both the X's', work out concretely, "
    "from the message and the anchors shown, exactly which rooms that refers "
    "to — do not attach the fact to every room anchor, only the ones meant.\n\n"
    "Example:\n"
    "Known anchors:\n- project (project): Project\n- room:8f3a1c2d (room): Kitchen\n\n"
    "Recently mentioned items:\n(none recent)\n\n"
    "Recent relations:\n(none recent)\n\n"
    "New message: \"acrylic finish on the kitchen cabinets\"\n"
    "-> new_nodes: [{id: kitchen_cabinet, label: 'Kitchen cabinet', type: entity}, "
    "{id: kitchen_cabinet_acrylic, label: 'Acrylic finish', type: attribute}]\n"
    "-> new_edges: [{source: kitchen_cabinet, target: 'room:8f3a1c2d', relation: "
    "located_in}, {source: kitchen_cabinet, target: kitchen_cabinet_acrylic, "
    "relation: uses_material}]\n\n"
    "Example (rejected alternative):\n"
    "New message: \"we're going with quartz instead of the marble countertop\"\n"
    "-> new_nodes: [{id: countertop_quartz, label: 'Quartz countertop', type: "
    "attribute}, {id: countertop_marble, label: 'Marble countertop (rejected)', "
    "type: attribute}]\n"
    "-> new_edges: [{source: countertop_marble, target: countertop_quartz, "
    "relation: rejected_in_favor_of}]\n\n"
    "Example (project total AND a separate room figure in one message):\n"
    "Known anchors:\n- project (project): Project\n- room:8f3a1c2d (room): Kitchen\n\n"
    "Recently mentioned items:\n(none recent)\n\n"
    "Recent relations:\n(none recent)\n\n"
    "New message: \"total budget is 15 lakh, and 4 lakh of that is for the kitchen\"\n"
    "-> new_nodes: [{id: project_budget_total, label: '15 lakh total budget', "
    "type: attribute}, {id: kitchen_budget, label: '4 lakh kitchen budget', "
    "type: attribute}]\n"
    "-> new_edges: [{source: project_budget_total, target: project, relation: "
    "budget_for}, {source: kitchen_budget, target: 'room:8f3a1c2d', relation: "
    "budget_for}]\n"
    "(TWO separate nodes, each with its own edge to its own target — never "
    "one node claimed by both, and the budget node is always the edge's "
    "source, the thing it applies to is always the target.)\n\n"
    "Example (correction WITHOUT 'instead' phrasing — match this pattern, "
    "not just explicit contrast phrasing):\n"
    "Candidates:\n- attr_a1b2c3d4 (attribute): Acrylic finish\n\n"
    "New message: \"actually let's do a matte lacquer on those cabinets\"\n"
    "-> revised_nodes: [{target_node_id: attr_a1b2c3d4, new_label: 'Matte "
    "lacquer finish'}]\n"
    "-> new_nodes: [], new_edges: []\n\n"
    "Example (retraction):\n"
    "Candidates:\n- attr_accent_wall (attribute): Navy accent wall\n\n"
    "New message: \"never mind the accent wall\"\n"
    "-> retracted_node_ids: [attr_accent_wall]"
)


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
    anchors_summary = "\n".join(f"- {a['id']} ({a['type']}): {a['label']}" for a in anchors) or "(none yet)"
    nodes_summary = "\n".join(f"- {n['id']} ({n['type']}): {n['label']}" for n in recent_nodes) or "(none recent)"
    edges_summary = (
        "\n".join(f"- {e['source']} -[{e['relation']}]-> {e['target']}" for e in recent_edges) or "(none recent)"
    )
    messages = [
        {"role": "system", "content": _GRAPH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Known anchors (the project and its rooms/budgets — always exist, "
                f"reference by id):\n{anchors_summary}\n\n"
                f"Recently mentioned items:\n{nodes_summary}\n\n"
                f"Recent relations:\n{edges_summary}\n\n"
                f"New message: {message}"
            ),
        },
    ]
    if capture is not None:
        capture["messages"] = messages
    result, completion = await structured_client.chat.completions.create_with_completion(
        model=settings.model_extraction,
        response_model=GraphExtraction,
        max_tokens=1024,
        max_retries=max_retries,
        messages=messages,
    )
    if capture is not None:
        capture["raw_output"] = _raw_completion_text(completion)
    return result


_QUESTION_PERSONA = (
    "You are a senior interior designer, briefing a junior designer who is "
    "gathering client details for a project quotation. Ask the way one designer "
    "asks a colleague in a normal working conversation — natural, warm, plain "
    "language — never like a form field or questionnaire prompt.\n\n"
    "Stay neutral: ask the question and nothing more. Do not volunteer your own "
    "opinion, a recommendation, or a suggested budget/material figure unless the "
    "question itself is explicitly asking the junior designer whether they'd "
    "like a suggestion. Do not comment on, second-guess, or react to anything "
    "already given — just move the intake forward."
)


async def generate_question(
    field_name: str,
    context: dict,
    *,
    is_retry: bool = False,
    capture: dict | None = None,
) -> str:
    if field_name == "materials":
        task = (
            "Ask ONE general, open-ended question about material preferences for "
            "the room, tailored to the roomType already known (e.g. for a kitchen "
            "you might mention countertops, cabinets, flooring as examples; for a "
            "living room, flooring or furniture). Give examples loosely, don't "
            "demand a checklist or ask about each surface separately — just invite "
            "whatever materials come to mind."
        )
    else:
        task = f"Ask one concise, natural question that gathers the '{field_name}' field."

    if is_retry:
        framing = (
            "The junior designer didn't have an answer last time this was asked. "
            "Ask again, gently — keep it close to how it was likely asked before, "
            "don't add a new example or a different framing, and make clear it's "
            "fine if they still don't have that detail."
        )
    else:
        framing = ""

    messages = [
        {
            "role": "system",
            "content": f"{_QUESTION_PERSONA}\n\n{task}" + (f"\n\n{framing}" if framing else ""),
        },
        {
            "role": "user",
            "content": f"Known so far: {context}",
        },
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
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


_WRAPUP_PERSONA = (
    "You are a senior interior designer, briefing a junior designer who just "
    "finished gathering client details for a project quotation. Everything "
    "needed has been captured (any leftover details were filled in with "
    "reasonable assumptions, not asked about again). Write ONE short, warm "
    "closing line for the junior designer to say to the client — a plain "
    "declarative statement, NOT a question, and don't propose or hint at "
    "asking anything further. Natural, professional, no exclamation-mark "
    "enthusiasm."
)


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
        {"role": "system", "content": _WRAPUP_PERSONA},
        {"role": "user", "content": f"Project details gathered: {context}"},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


_MERGE_PERSONA = (
    "You are a senior interior designer, briefing a junior designer, composing ONE short reply "
    "that covers everything from this turn (a fact just noted, a question just answered, or "
    "both) as a single natural message — never a list, never labeled sections, never repeating "
    "a piece verbatim. If multiple things happened this turn, blend them the way a person "
    "actually talks, not a bulleted summary."
)


async def merge_response(parts: list[str], *, capture: dict | None = None) -> str:
    """Used by app.execution.merge_task_results (Phase 11) to compose several
    deterministic/generated pieces from one multi-task turn into one reply.
    Reuses model_question_gen (fast Turbo model), same latency reasoning as
    generate_wrapup_message — this is short, low-stakes composition, not a
    real answer needing the slower model."""
    messages = [
        {"role": "system", "content": _MERGE_PERSONA},
        {"role": "user", "content": "Pieces to blend into one reply:\n" + "\n---\n".join(parts)},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


_CONFIRM_CHANGE_PERSONA = (
    "You are a senior interior designer, briefing a junior designer, about to change a client "
    "detail that was already recorded. Ask ONE short, natural confirmation question — mention "
    "the old value and the new one plainly (not a form-style diff), and ask whether the change "
    "is really what the client meant. Neutral tone, no assumption either way."
)


async def generate_conflict_confirmation(
    field_label: str, old_value, new_value, *, capture: dict | None = None
) -> str:
    """Used by app.graph.build_context_node when a critical-tier field's
    value would change (app.context_builder.detect_conflicts) — same call
    shape as generate_wrapup_message/merge_response (fast model_question_gen,
    short/low-stakes), not a real answer needing the slower model."""
    messages = [
        {"role": "system", "content": _CONFIRM_CHANGE_PERSONA},
        {"role": "user", "content": f"Field: {field_label}\nPreviously recorded: {old_value}\nJust stated: {new_value}"},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


_CONFIRM_DELETE_PERSONA = (
    "You are a senior interior designer, briefing a junior designer, about to remove a client "
    "detail that was already recorded. Ask ONE short, natural confirmation question — name the "
    "thing being removed plainly, and ask whether the client really wants it taken out. Neutral "
    "tone, no assumption either way — this is a removal, not a value change, so never phrase it "
    "as 'change X from A to B'."
)


async def generate_conflict_confirmation_delete(old_value, *, capture: dict | None = None) -> str:
    """Delete's counterpart to generate_conflict_confirmation above — a
    separate function rather than an optional-everything signature on that
    one, since "remove X?" and "change X from A to B?" are different enough
    prompts to want their own persona text. Used by
    app.graph.delete_context_node for a critical-tier removal; same
    fast/low-stakes model_question_gen call shape as every other confirmation/
    question-gen call in this module."""
    messages = [
        {"role": "system", "content": _CONFIRM_DELETE_PERSONA},
        {"role": "user", "content": f"Currently recorded: {old_value}\nClient asked to remove this."},
    ]
    resp = await client.chat.completions.create(
        model=settings.model_question_gen,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return resp.choices[0].message.content or ""


async def infer_missing_field(field_name: str, context: dict, *, capture: dict | None = None) -> str:
    """Terminal fallback for a field the client's retry budget ran out on
    before they answered (see app.graph.decline_field_node, which calls this
    via app.inference.infer_field the moment a field's attempts are
    exhausted) — produces a single reasonable value grounded in whatever's
    already known (room type, style, budget, typical ranges), so the project
    can complete instead of blocking indefinitely. The caller stores this
    with changed_by="inferred", not "user_message"."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior interior designer filling in a single missing "
                "quotation detail with your best professional estimate, because the "
                "client's answer wasn't available. Given the project details already "
                "known, output ONLY a short, concrete value for the requested field — "
                "a typical/reasonable figure or description, grounded in the known "
                "context (not a generic placeholder). No explanation, no caveats, just "
                "the value itself (e.g. '150 sqft', '$8k-$12k', 'modern')."
            ),
        },
        {
            "role": "user",
            "content": f"Known project details: {context}\n\nField to estimate: {field_name}",
        },
    ]
    resp = await client.chat.completions.create(
        model=settings.model_extraction,
        messages=messages,
        temperature=0.3,
        max_tokens=64,
    )
    if capture is not None:
        capture["messages"] = messages
        capture["raw_output"] = resp.choices[0].message.content
    return (resp.choices[0].message.content or "").strip()


async def generate_answer(
    message: str, context: dict, history: str = "", retrieved: str = "", *, capture: dict | None = None
) -> AsyncIterator[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior interior designer answering a colleague's question "
                "in the middle of a client intake. Use the project context and any "
                "retrieved reference material to answer helpfully and concisely, in a "
                "natural, professional voice — not a form or a lecture."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Project context: {context}\n\n"
                f"Retrieved references: {retrieved or 'none'}\n\n"
                f"Conversation so far:\n{history}\n\n"
                f"User: {message}"
            ),
        },
    ]
    if capture is not None:
        capture["messages"] = messages
    stream = await client.chat.completions.create(
        model=settings.model_answer,
        messages=messages,
        stream=True,
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
