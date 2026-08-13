"""Knowledge-Gap Question Generation — Phase 15. Replaces
PartialContext.next_field_to_ask()'s fixed PROJECT_FIELDS/ROOM_FIELDS
priority chain with a walk over ontology/v1.yaml itself — any
ontology-defined field with no confirmed KnowledgeNode value gets asked
about, full stop, with no separately-maintained field list that can fall
out of sync with the ontology (the actual fix for the "timeline needed its
own end-of-function special case" bug class the plan describes — that bug
was really "the walked list and the ontology can diverge," and there's now
only one list: the ontology itself). This now also covers the room field
list itself (_ROOM_FIELD_NODE_TYPES, read from ontology/v1.yaml's
Rooms.fields) — it used to be a hand-maintained mirror of that list.

Scope: ports next_field_to_ask()'s job for the STRUCTURED fields
(BasicInformation, Budget, Timeline, and each room's RoomType/Budget/Style/
SquareFootage/ExistingFurniture). The plan's fuller
find_knowledge_gap(project_id, current_task) — walking exactly what a
specific task like QUOTE_GENERATION needs — isn't buildable yet:
QUOTE_GENERATION was never added to app.tasks.TaskType (see that module's
docstring — SAVE_CONTEXT/DELETE_CONTEXT/MEMORY/QUOTE_GENERATION are all
deliberately unbuilt, nothing implements them, so there's no per-task
dependency subtree to walk). This module answers "what's still missing for
this PROJECT overall" — a real, useful subset of Phase 15's goal.

moreRoomsPending is deliberately not walked here — Phase 3 already decided
it's a dialogue-mechanic flag, not a KnowledgeNode fact (out of ontology
entirely — ontology/PHASE3_ONTOLOGY.md). Not asking about it isn't a new
gap, it's that same decision holding.

Retry-budget tiering (rephrase-then-infer on decline) has been removed —
decline is now a whole-room skip (app.graph.classify_intent_node), not a
per-field retry count, so KnowledgeGap no longer carries a tier and this
module no longer needs FIELD_TIERS.

find_knowledge_gaps() returns a KnowledgeGapBatch — every open field for one
room at once, not one field per turn — see app.graph.generate_question_node,
which is what actually wires this into the live turn (this module itself
stays DB-adjacent but pure of any graph-node routing decisions)."""

from typing import Optional

from pydantic import BaseModel, Field

from app import llm, graph_store
from app.canonical_mapper import _ONTOLOGY
from app.context_builder import LEAF_TO_FIELD_NAME
from app.models import FIELD_LABELS

# Room field node types, in walk order — read straight from the ontology
# (ontology/v1.yaml's Rooms.fields) instead of a hand-maintained Python list,
# so a future ontology change (more/fewer room fields) is picked up here with
# no code change. Labels come from context_builder.LEAF_TO_FIELD_NAME, the
# same node_type -> field_name table app/graph.py already uses for question
# labels/confirmation summaries — one source of truth, not a second copy.
_ROOM_FIELD_NODE_TYPES: list[str] = _ONTOLOGY["Rooms"]["fields"]

# Sentinel path for "no room exists yet" — mirrors PartialContext.next_field_to_ask()'s
# own "room:new.roomType" sentinel key. Not a real KnowledgeNode path.
_NEW_ROOM_PATH = "Project.Rooms.<new>"


class KnowledgeGap(BaseModel):
    canonical_path: str
    field_label: str
    node_type: str
    room_id: Optional[str] = None


class KnowledgeGapBatch(BaseModel):
    """What generate_question_node actually asks about on a given turn — one
    or more KnowledgeGaps to fold into a single combined question. `room_id`
    is set whenever this batch is every open room field for one room (see
    find_knowledge_gaps); it's None for the project-level blocking gaps
    (ProjectType, room existence, Timeline, Budget.Total), which stay
    single-item batches — batching those isn't the "one field at a time per
    room" problem this exists to fix."""

    gaps: list[KnowledgeGap] = Field(default_factory=list)
    room_id: Optional[str] = None


async def find_knowledge_gaps(
    project_id: str, active_room_id: Optional[str] = None, skipped_rooms: Optional[list[str]] = None
) -> Optional[KnowledgeGapBatch]:
    """The turn's whole "what's still missing" answer, batched per scope
    instead of one field at a time. Room fields are the actual fix here: all
    of a room's open fields (per _ROOM_FIELD_NODE_TYPES) come back together
    as one KnowledgeGapBatch instead of forcing a separate turn per field —
    see app.graph.generate_question_node, which folds a batch's gaps into one
    combined question. The three project-level blocking gaps (ProjectType,
    room existence, and the Timeline/Budget.Total pair at the very end) stay
    single-item batches; they aren't the "one field at a time per room"
    problem this exists to fix.

    `active_room_id` is which room's batch to prefer (see
    app.graph.classify_intent_node's connection-derived tracking) — it stays
    in focus as long as it still has something open. Once it doesn't, this
    auto-advances to the next room with anything open (same priority order
    the old walk used: non-skipped rooms oldest-first, then skipped rooms) —
    the caller is expected to persist whatever room_id comes back as the new
    active_room_id, since this function is also what decides when focus
    shifts forward."""
    # lifecycle == "active" filter: a retracted node (app.graph.delete_context_node)
    # must count as "open" again, not "known" — excluding it here (rather than
    # special-casing every lookup below) covers both _is_open AND the room_ids
    # comprehension in one place, so a retracted room's own container node
    # can't keep the room "known" once its RoomType leaf is gone too.
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    by_path = {n.canonical_path: n for n in nodes}
    skipped = set(skipped_rooms or [])

    def _is_open(path: str) -> bool:
        node = by_path.get(path)
        # Nothing in this pipeline sets status="skipped" yet (that's the
        # retry-exhaustion mechanic DialogueState/Phase 7's still-deferred
        # cutover owns) — a missing value is the only "open" signal today.
        return node is None or node.value is None

    if _is_open("Project.BasicInformation.ProjectType"):
        return KnowledgeGapBatch(
            gaps=[
                KnowledgeGap(
                    canonical_path="Project.BasicInformation.ProjectType",
                    field_label=FIELD_LABELS.get("projectType", "project type"),
                    node_type="ProjectType",
                )
            ]
        )

    # Room container nodes, oldest first — see app.graph.classify_intent_node's
    # decline-detection/skipped_rooms mechanism (a room only ever moves INTO
    # skipped_rooms, never explicitly out — see the two-pass walk below for
    # why no explicit "revisit" trigger is needed).
    room_containers = sorted(
        (n for n in nodes if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms" and n.room_id),
        key=lambda n: n.created_at,
    )
    room_ids = [n.room_id for n in room_containers]
    if not room_ids:
        return KnowledgeGapBatch(
            gaps=[
                KnowledgeGap(
                    canonical_path=_NEW_ROOM_PATH, field_label=FIELD_LABELS.get("roomType", "room type"),
                    node_type="RoomType",
                )
            ]
        )

    def _room_gaps(room_id: str) -> list[KnowledgeGap]:
        gaps = []
        for node_type in _ROOM_FIELD_NODE_TYPES:
            path = f"Project.Rooms.{room_id}.{node_type}"
            if _is_open(path):
                field_name = LEAF_TO_FIELD_NAME.get(node_type, node_type)
                gaps.append(
                    KnowledgeGap(
                        canonical_path=path, field_label=FIELD_LABELS.get(field_name, field_name),
                        node_type=node_type, room_id=room_id,
                    )
                )
        return gaps

    if active_room_id and active_room_id in room_ids:
        active_gaps = _room_gaps(active_room_id)
        if active_gaps:
            return KnowledgeGapBatch(gaps=active_gaps, room_id=active_room_id)

    # Active room has nothing left open (or there wasn't one yet) —
    # auto-advance: first room with anything open, non-skipped rooms before
    # skipped ones, oldest first within each group. A room is never
    # explicitly "unskipped" — once every non-skipped room is exhausted,
    # skipped rooms simply become the only candidates left in the second
    # pass, same standing low-priority tier as before.
    non_skipped = [r for r in room_ids if r not in skipped]
    skipped_ordered = [r for r in room_ids if r in skipped]
    for candidate_rooms in (non_skipped, skipped_ordered):
        for room_id in candidate_rooms:
            gaps = _room_gaps(room_id)
            if gaps:
                return KnowledgeGapBatch(gaps=gaps, room_id=room_id)

    if _is_open("Project.Timeline.Value"):
        return KnowledgeGapBatch(
            gaps=[
                KnowledgeGap(
                    canonical_path="Project.Timeline.Value", field_label=FIELD_LABELS.get("timeline", "timeline"),
                    node_type="Value",
                )
            ]
        )

    if _is_open("Project.Budget.Total"):
        return KnowledgeGapBatch(
            gaps=[
                KnowledgeGap(
                    canonical_path="Project.Budget.Total", field_label=FIELD_LABELS.get("overallBudget", "overall budget"),
                    node_type="Total",
                )
            ]
        )

    return None


async def generate_question(
    gaps: list[KnowledgeGap], context: dict, *, is_retry: bool = False, capture: dict | None = None
) -> str:
    """Thin adapter onto the existing llm.generate_question call — same
    model, just fed every gap in the batch's field_label at once so the LLM
    phrases one combined question instead of one call per field."""
    return await llm.generate_question([g.field_label for g in gaps], context, is_retry=is_retry, capture=capture)
