from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Literal, Optional
from uuid import uuid4

from beanie import Document
from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class TraceEntry(BaseModel):
    node_name: str
    model_used: Optional[str] = None
    input_summary: str
    output_summary: str
    duration_ms: int
    timestamp: datetime = Field(default_factory=utcnow)
    # Full, untruncated prompt/response for steps that called an LLM — None
    # for rule-based steps, which is how the viewer tells the two apart.
    llm_input: Optional[list[dict]] = None
    llm_output: Optional[str] = None


# Every field name below appears in exactly one tier, which drives retry
# behavior in app/graph.py: critical fields get up to 3 rephrases before being
# given up on, moderate fields get 1, optional fields are asked once and never
# rephrased. project-level and room-level field names intentionally share this
# one table since a field name is unique across the two scopes.
FIELD_TIERS: dict[str, Literal["critical", "moderate", "optional"]] = {
    "projectType": "critical",
    "overallBudget": "critical",
    "moreRoomsPending": "critical",
    "roomType": "critical",
    "budgetOrRequirement": "critical",
    "style": "moderate",
    "squareFootage": "moderate",
    "timeline": "moderate",
    "existingFurniture": "optional",
    "materials": "optional",
}

# Rephrases allowed AFTER the first ask — e.g. "critical": 3 means up to 4
# total asks (1 original + 3 rephrases) before the field is marked "skipped".
RETRY_LIMITS: dict[str, int] = {"critical": 3, "moderate": 1, "optional": 0}

# Human-readable form of a field name — used anywhere a field needs to be
# named in a message shown to a user or fed into an LLM prompt (confirmation
# summaries, classify_intent's pending-question context, save_project's
# assumption notes). Shared across app/graph.py and app/understanding.py, so
# it lives here rather than in either.
FIELD_LABELS: dict[str, str] = {
    "projectType": "project type",
    "overallBudget": "overall budget",
    "timeline": "timeline",
    "moreRoomsPending": "more rooms",
    "roomType": "room type",
    "budgetOrRequirement": "budget",
    "style": "style",
    "squareFootage": "square footage",
    "existingFurniture": "existing furniture",
}

FieldStatus = Literal["confirmed", "assumed", "skipped"]


class ProjectMeta(BaseModel):
    """Project-level fields — apply once per quotation, not per room."""

    projectType: Optional[str] = None  # "new construction" | "renovation" | "refresh" | free text
    overallBudget: Optional[str] = None
    timeline: Optional[str] = None
    # None = not yet asked whether more rooms are in scope; True/False once asked.
    moreRoomsPending: Optional[bool] = None


class RoomContext(BaseModel):
    """One room/space within a project. A project can hold several of these."""

    room_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    roomType: Optional[str] = None  # a suggested vocabulary term, or free text via "other"
    budgetOrRequirement: Optional[str] = None  # a number OR a description satisfies this
    style: Optional[str] = None
    squareFootage: Optional[float] = None
    existingFurniture: Optional[str] = None
    materials: Optional[list[MaterialSpec]] = None


# A one-character-off typo on a short room name ("bedrroom" vs "bedroom")
# scores well above this; two genuinely different room types ("kitchen" vs
# "bedroom") score well below it. Shared by app.context_builder._resolve_room
# and app.canonical_mapper's room-scoping so every path agrees on what counts
# as "the same room" — live-observed without this (in the pre-cutover
# PartialContext.resolve_room this originally lived on): a later turn's
# correctly-spelled respelling of an already-known room silently created a
# duplicate room instead of updating the existing one.
_ROOM_TYPE_FUZZY_MATCH_THRESHOLD = 0.82


def room_type_matches(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= _ROOM_TYPE_FUZZY_MATCH_THRESHOLD


class ChatSession(Document):
    """Live conversation state. As of the KnowledgeNode cutover (see
    ARCHITECTURE_BASELINE.md), this holds only dialogue mechanics —
    project facts themselves live in KnowledgeNode, keyed by project_id."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    # Minted at session creation (not at completion, unlike the old
    # ProjectContext.project_id) — every KnowledgeNode this session ever
    # writes uses this as its project_id from turn 1.
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = Field(default_factory=list)
    trace: list[TraceEntry] = Field(default_factory=list)
    status: Literal["in_progress", "complete"] = "in_progress"
    active_room_id: Optional[str] = None
    # Retry counters for declined questions, keyed by canonical_path — see
    # app.graph.decline_field_node. Dialogue mechanics, not a project fact,
    # so it lives here rather than on a KnowledgeNode.
    field_attempts: dict[str, int] = Field(default_factory=dict)
    # Serialized app.question_engine.KnowledgeGap for the currently pending
    # question, if any — lets the next turn attribute a decline to the right
    # field (and know its tier) without re-deriving it.
    pending_gap: Optional[dict] = None
    # Serialized app.context_builder.Conflict awaiting a yes/no reply — set
    # when a critical-tier field's value would change; see app.graph's
    # confirm_conflict_node.
    pending_confirmation: Optional[dict] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "sessions"


class ProjectContext(Document):
    """A completion-time snapshot, materialized from the KnowledgeNode tree
    (see app.graph.complete_project_node) — not the live source of truth,
    which is KnowledgeNode itself under this project_id."""

    project_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    # Flat snapshot of the project's structured fields at completion —
    # {"projectType": ..., "rooms": [{"roomType": ..., ...}, ...], ...} —
    # not a typed model, since the shape is just whatever
    # materialize_project_summary read off KnowledgeNode.
    summary: dict[str, Any] = Field(default_factory=dict)
    # Human-readable notes on which fields were system-inferred/calculated
    # rather than user-stated, e.g. "kitchen squareFootage: assumed 150 sqft"
    # — built from app.facts/inference/calculation's provenance queries, not
    # a manually-accumulated list. Surfaced so the quotation is honest about
    # what's firm vs. estimated.
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "projects"


# Provenance of a value in KnowledgeNodeVersion (Phase 8) — see that class
# and app/versioning.py. Named here (not inline) so app/context_builder.py's
# ProposedWrite.changed_by can share the exact same type.
ChangedBy = Literal["user_message", "inferred", "system_default"]


class KnowledgeNode(Document):
    """One node in the canonical knowledge tree — see ontology/v1.yaml and
    ontology/PHASE3_ONTOLOGY.md for the schema canonical_path values are
    drawn from. The live source of truth for every project fact — written
    directly by the chat turn (app.graph.build_context_node, via
    app.context_builder/app.canonical_mapper), not PartialContext/ContextGraph,
    which this replaced (see ARCHITECTURE_BASELINE.md).

    node_id (not `id`) is the business key parent_id/children_ids reference —
    same "leave Beanie's own `id`/ObjectId alone, add a separate uuid business
    key" pattern ChatSession.session_id/ProjectContext.project_id/
    CatalogItem.item_id already use, kept for consistency rather than
    overriding `id` itself as the plan's own sketch does."""

    node_id: str = Field(default_factory=lambda: uuid4().hex)
    canonical_path: str
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    # Matches a node type name in ontology/v1.yaml, e.g. "Rooms", "Materials",
    # "ProjectType".
    node_type: str
    value: Optional[Any] = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    # "skipped" never actually lands here: app.graph.decline_field_node calls
    # app.inference.infer_field the moment a field's retry budget is
    # exhausted, so a node's status is always "confirmed" or "assumed".
    status: FieldStatus = "confirmed"
    # Deletion lifecycle (app.graph.delete_context_node / app.versioning.retract_node)
    # — deliberately a separate field from `status` above, which tracks
    # confirmed-vs-assumed provenance, not whether a fact still holds. Two
    # states, not three: apply_to_graph always mutates a node's `value` in
    # place on edit (history lives in knowledge_node_versions instead), so
    # nothing in this codebase would ever produce a "superseded" node — see
    # the corresponding note in the deletion-support plan.
    lifecycle: Literal["active", "retracted"] = "active"
    # Denormalized from this node's latest KnowledgeNodeVersion row (Phase 8)
    # — same "fast read off the live node, full history in the append-only
    # log" split already used for `value` itself. Kept in sync by whatever
    # writes `value` (app.context_builder.apply_to_graph,
    # app.canonical_mapper._create_instance, app.dependency_graph.recompute_dependents)
    # — see app/facts.py, app/inference.py, app/calculation.py (Phase 12),
    # which query this field directly rather than walking history per node.
    changed_by: ChangedBy = "user_message"
    source_message_id: Optional[str] = None
    project_id: str
    room_id: Optional[str] = None
    # Set on a "Label" leaf (see app/canonical_mapper.py) so a repeat mapping
    # call can score against it directly instead of re-embedding the same
    # unchanging text every time. None for every other node_type, and for a
    # Label created before this field existed (e.g. by
    # scripts/migrate_partial_context_to_tree.py) — canonical_mapper backfills
    # it lazily the first time such a node is considered as a candidate.
    embedding: Optional[list[float]] = None
    # Unused today — this system has exactly one tenant. Added now, indexed
    # (see app/database.py), because retrofitting tenant scoping onto an
    # already-growing collection later is materially harder than shipping an
    # unused column today (Phase 6 of the implementation plan's own call).
    tenant_id: Optional[str] = None
    version: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "knowledge_nodes"


class KnowledgeNodeVersion(Document):
    """Append-only history of every value a KnowledgeNode has held — same
    shape/purpose as TraceEntry's existing append-only pattern for turn
    history above, just a separate collection since this outlives a single
    chat turn. Written only by app/versioning.py::record_version(), never
    mutated once inserted — see ontology/PHASE8_VERSIONING.md."""

    node_id: str
    version: int
    value: Optional[Any] = None
    changed_at: datetime = Field(default_factory=utcnow)
    # "user_message": stated directly by the client this turn. "inferred":
    # an LLM's best-guess fallback (app.deepinfra.infer_missing_field's role
    # today, not yet ported to this pipeline — see PHASE8_VERSIONING.md).
    # "system_default": a deterministic, non-LLM default. Only "user_message"
    # is actually produced by any code path as of Phase 8 — the other two
    # are real values Phase 12 (Fact/Inference/Calculation split) wires up,
    # not placeholders invented here.
    changed_by: ChangedBy = "user_message"
    source_message_id: Optional[str] = None

    class Settings:
        name = "knowledge_node_versions"


# GraphRelation (above) minus part_of/located_in — those are superseded by
# canonical_path containment once a fact lives in KnowledgeNode (Gap 4,
# ontology/PHASE3_ONTOLOGY.md) — plus "derives_from" for dependency edges
# (Phase 9), which ContextGraph never had a use for.
KnowledgeEdgeRelation = Literal[
    "derives_from", "uses_material", "applies_to", "modifies", "requires", "budget_for", "rejected_in_favor_of", "revises"
]


class KnowledgeEdge(Document):
    """A non-hierarchical relationship between two KnowledgeNodes — see
    ontology/PHASE9_DEPENDENCY_GRAPH.md. Containment ("this is in the
    kitchen") is never represented here; canonical_path already expresses
    it. New collection (knowledge_edges), not yet written by any live turn —
    see app/dependency_graph.py."""

    edge_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str  # KnowledgeNode.node_id
    target_id: str  # KnowledgeNode.node_id
    relation: KnowledgeEdgeRelation
    project_id: str
    created_at: datetime = Field(default_factory=utcnow)

    class Settings:
        name = "knowledge_edges"


class CatalogItem(Document):
    item_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    style_tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)

    class Settings:
        name = "catalog"
