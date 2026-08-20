from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from beanie import Document
from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # Only ever set on an assistant message produced by
    # app.llm.generate_turn_reply's join step — the markdown tables that
    # accompany `content` (the conversational reply) for that same turn, kept
    # as their own fields rather than concatenated into `content` so the
    # viewer can render/style them separately. None for every user message
    # and for an assistant message with nothing to report (a plain reply/
    # question with no changes or context this turn).
    context_summary: Optional[str] = None
    changes_summary: Optional[str] = None
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

# Human-readable form of a field name — used anywhere a field needs to be
# named in a message shown to a user or fed into an LLM prompt (confirmation
# summaries, classify_operations' pending-question context, save_project's
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
    # Rooms currently deprioritized by a decline — see
    # app.graph.classify_intent_node's decline-detection step and
    # app.question_engine.find_knowledge_gap's two-pass breadth-first walk.
    skipped_rooms: list[str] = Field(default_factory=list)
    # ProjectType is asked at most once, not held as a blocking gap — only
    # room existence is truly mandatory. Set True by app.pipeline.run_pipeline
    # the turn after ProjectType was the pending question, whether or not the
    # user actually answered it (see app.question_engine.find_knowledge_gaps).
    project_type_skipped: bool = False
    # {"canonical_path", "room_id"} for the field the LAST question was
    # about — written only by app.graph.generate_question_node, read only by
    # classify_intent_node. Dialogue mechanics, not a project fact, so it
    # lives here rather than on a KnowledgeNode.
    current_field: Optional[dict] = None
    # Serialized app.question_engine.KnowledgeGapBatch for the currently open
    # gap(s) — a room's whole open-field batch, or a single project-level
    # gap. Written by build_context_node right after a value commits, and by
    # generate_question_node right after it computes the batch it's about to
    # ask about — no other node reads or writes this. See PENDING_GAP_ANALYSIS.md.
    pending_gap: Optional[dict] = None
    # Which room a batched question/write is currently focused on — a
    # narrower, newer concept than the old removed active_room_id (see
    # PENDING_GAP_ANALYSIS.md §7): this one drives ONLY which room's open
    # fields get batched into the next question, never where a write lands
    # (that's still exclusively task.connection/room_hint). Set by
    # app.graph.classify_intent_node from the resolved room of the LAST
    # EDIT_CONTEXT/DELETE_CONTEXT/RETRIEVE_CONTEXT task in a turn, and
    # auto-advanced by app.question_engine.find_knowledge_gaps once the
    # current active room has nothing left open.
    active_room_id: Optional[str] = None
    # Set when classify_operations returned a write task with connection=
    # None — one clarifying question per unresolved operation, answered
    # together via the next request's ChatRequest.operation_answers. See
    # app.graph.classify_intent_node / GraphState.pending_operation_questions.
    pending_operation_questions: Optional[dict] = None
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


class KnowledgeNode(BaseModel):
    """One node in the canonical knowledge tree — see ontology/v1.yaml and
    ontology/PHASE3_ONTOLOGY.md for the schema canonical_path values are
    drawn from. The live source of truth for every project fact — written
    directly by the chat turn (app.graph.build_context_node, via
    app.context_builder/app.canonical_mapper), not PartialContext/ContextGraph,
    which this replaced (see ARCHITECTURE_BASELINE.md).

    Lives in Neo4j, not Mongo (see app/graph_store.py) — this is a plain
    Pydantic DTO hydrated from a Cypher result, not a Beanie Document. A
    :KNode node's own graph-native properties are exactly this model's
    fields minus `parent_id`, which is derived per read from the node's
    outgoing :CHILD_OF relationship rather than stored as a property (a
    real edge, not a denormalized string, is the whole point of the move).
    There's no `children_ids` field either — walk :CHILD_OF the other
    direction (see graph_store.descendant_ids) instead of maintaining a
    parallel list that could drift from the actual relationships.

    node_id (not `id`) is the business key other nodes' CHILD_OF edges and
    KnowledgeEdge rows reference — same pattern
    ChatSession.session_id/ProjectContext.project_id/CatalogItem.item_id
    already use."""

    node_id: str = Field(default_factory=lambda: uuid4().hex)
    canonical_path: str
    parent_id: Optional[str] = None
    # Matches a node type name in ontology/v1.yaml, e.g. "Rooms", "Materials",
    # "ProjectType".
    node_type: str
    value: Optional[Any] = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    # "skipped" never actually lands here — a declined room-scoped field is
    # tracked via ChatSession.skipped_rooms instead (see
    # app.graph.classify_intent_node's decline-detection), not on the node
    # itself, so a node's status is always "confirmed" or "assumed".
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
    # (see app/neo4j_db.py), because retrofitting tenant scoping onto an
    # already-growing graph later is materially harder than shipping an
    # unused property today (Phase 6 of the implementation plan's own call).
    tenant_id: Optional[str] = None
    version: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class KnowledgeNodeVersion(BaseModel):
    """Append-only history of every value a KnowledgeNode has held — same
    shape/purpose as TraceEntry's existing append-only pattern for turn
    history above, just a separate set of :KNodeVersion nodes in Neo4j since
    this outlives a single chat turn. Written only by
    app/versioning.py::record_version(), never mutated once inserted — see
    ontology/PHASE8_VERSIONING.md. No relationship back to its KnowledgeNode
    (unlike KNode's own :CHILD_OF edges) — every read path queries by
    `node_id` property directly (indexed, see app/neo4j_db.py), so a graph
    edge would add nothing a property lookup doesn't already give for free."""

    node_id: str
    version: int
    value: Optional[Any] = None
    changed_at: datetime = Field(default_factory=utcnow)
    # "user_message": stated directly by the client this turn. "inferred":
    # an LLM's best-guess fallback (app.llm.infer_missing_field's role
    # today, not yet ported to this pipeline — see PHASE8_VERSIONING.md).
    # "system_default": a deterministic, non-LLM default. Only "user_message"
    # is actually produced by any code path as of Phase 8 — the other two
    # are real values Phase 12 (Fact/Inference/Calculation split) wires up,
    # not placeholders invented here.
    changed_by: ChangedBy = "user_message"
    source_message_id: Optional[str] = None


# GraphRelation (above) minus part_of/located_in — those are superseded by
# canonical_path containment once a fact lives in KnowledgeNode (Gap 4,
# ontology/PHASE3_ONTOLOGY.md) — plus "derives_from" for dependency edges
# (Phase 9), which ContextGraph never had a use for.
KnowledgeEdgeRelation = Literal[
    "derives_from", "uses_material", "applies_to", "modifies", "requires", "budget_for", "rejected_in_favor_of", "revises"
]


class KnowledgeEdge(BaseModel):
    """A non-hierarchical relationship between two KnowledgeNodes — see
    ontology/PHASE9_DEPENDENCY_GRAPH.md. Containment ("this is in the
    kitchen") is never represented here; canonical_path already expresses
    it. Stored as a real Neo4j relationship (a single generic `:REL` type
    carrying `relation` as a property, rather than one Neo4j relationship
    type per KnowledgeEdgeRelation value — see app/graph_store.py for why),
    not yet written by any live turn — see app/dependency_graph.py."""

    edge_id: str = Field(default_factory=lambda: uuid4().hex)
    source_id: str  # KnowledgeNode.node_id
    target_id: str  # KnowledgeNode.node_id
    relation: KnowledgeEdgeRelation
    project_id: str
    created_at: datetime = Field(default_factory=utcnow)


class CatalogItem(Document):
    item_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    style_tags: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)

    class Settings:
        name = "catalog"
