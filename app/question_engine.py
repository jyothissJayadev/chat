"""Knowledge-Gap Question Generation — Phase 15. Replaces
PartialContext.next_field_to_ask()'s fixed PROJECT_FIELDS/ROOM_FIELDS
priority chain with a walk over ontology/v1.yaml itself — any
ontology-defined field with no confirmed KnowledgeNode value gets asked
about, full stop, with no separately-maintained field list that can fall
out of sync with the ontology (the actual fix for the "timeline needed its
own end-of-function special case" bug class the plan describes — that bug
was really "the walked list and the ontology can diverge," and there's now
only one list: the ontology itself).

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

Retry-budget tiering (the product decision the plan flags: does tier depend
on ontology position, or on how many pending tasks need the field?) is
ontology-position-based — reusing FIELD_TIERS via the same leaf-node_type
mapping app.context_builder already established. The alternative
(pending-task-count-based) has nothing to be based on: there's no
multi-task dependency system for the unbuilt task types above, so ontology
position is the only option that's actually implementable today, not a
fresh judgment call — see ontology/PHASE15_KNOWLEDGE_GAP.md.

Not wired into the live turn — same standing as every module since Phase 1."""

from typing import Optional

from pydantic import BaseModel

from app import deepinfra
from app.models import FIELD_LABELS, FIELD_TIERS, KnowledgeNode

# Mirrors PartialContext.ROOM_FIELDS' order exactly (app/models.py) — same
# priority, just walked against KnowledgeNode instead of RoomContext.
_ROOM_FIELD_ORDER = [
    ("RoomType", "roomType"),
    ("Budget", "budgetOrRequirement"),
    ("Style", "style"),
    ("SquareFootage", "squareFootage"),
    ("ExistingFurniture", "existingFurniture"),
]

# Sentinel path for "no room exists yet" — mirrors PartialContext.next_field_to_ask()'s
# own "room:new.roomType" sentinel key. Not a real KnowledgeNode path.
_NEW_ROOM_PATH = "Project.Rooms.<new>"


class KnowledgeGap(BaseModel):
    canonical_path: str
    field_label: str
    node_type: str
    room_id: Optional[str] = None
    tier: str


async def find_knowledge_gap(project_id: str, active_room_id: Optional[str] = None) -> Optional[KnowledgeGap]:
    # lifecycle == "active" filter: a retracted node (app.graph.delete_context_node)
    # must count as "open" again, not "known" — excluding it here (rather than
    # special-casing every lookup below) covers both _is_open AND the room_ids
    # comprehension in one place, so a retracted room's own container node
    # can't keep the room "known" once its RoomType leaf is gone too.
    nodes = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.lifecycle == "active"
    ).to_list()
    by_path = {n.canonical_path: n for n in nodes}

    def _is_open(path: str) -> bool:
        node = by_path.get(path)
        # Nothing in this pipeline sets status="skipped" yet (that's the
        # retry-exhaustion mechanic DialogueState/Phase 7's still-deferred
        # cutover owns) — a missing value is the only "open" signal today.
        return node is None or node.value is None

    if _is_open("Project.BasicInformation.ProjectType"):
        return KnowledgeGap(
            canonical_path="Project.BasicInformation.ProjectType", field_label=FIELD_LABELS.get("projectType", "project type"),
            node_type="ProjectType", tier=FIELD_TIERS.get("projectType", "critical"),
        )
    if _is_open("Project.Budget.Total"):
        return KnowledgeGap(
            canonical_path="Project.Budget.Total", field_label=FIELD_LABELS.get("overallBudget", "overall budget"),
            node_type="Total", tier=FIELD_TIERS.get("overallBudget", "critical"),
        )

    room_ids = {n.room_id for n in nodes if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms" and n.room_id}
    if not room_ids:
        return KnowledgeGap(
            canonical_path=_NEW_ROOM_PATH, field_label=FIELD_LABELS.get("roomType", "room type"),
            node_type="RoomType", tier=FIELD_TIERS.get("roomType", "critical"),
        )

    room_id = active_room_id if active_room_id in room_ids else next(iter(room_ids))
    for node_type, field_name in _ROOM_FIELD_ORDER:
        path = f"Project.Rooms.{room_id}.{node_type}"
        if _is_open(path):
            return KnowledgeGap(
                canonical_path=path, field_label=FIELD_LABELS.get(field_name, field_name),
                node_type=node_type, room_id=room_id, tier=FIELD_TIERS.get(field_name, "moderate"),
            )

    if _is_open("Project.Timeline.Value"):
        return KnowledgeGap(
            canonical_path="Project.Timeline.Value", field_label=FIELD_LABELS.get("timeline", "timeline"),
            node_type="Value", tier=FIELD_TIERS.get("timeline", "moderate"),
        )

    return None


async def generate_question(gap: KnowledgeGap, context: dict, *, is_retry: bool = False, capture: dict | None = None) -> str:
    """Thin adapter onto the existing deepinfra.generate_question call — same
    model, same prompt shape, just fed KnowledgeGap.field_label instead of a
    raw PartialContext field-name string."""
    return await deepinfra.generate_question(gap.field_label, context, is_retry=is_retry, capture=capture)
