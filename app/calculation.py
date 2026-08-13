"""Calculation — Phase 12 (Fact/Inference/Calculation split). Pure
deterministic math, no LLM call at all — changed_by="system_default".

No actual calculation RULE lives here, same reasoning as
app.dependency_graph (Phase 9): no room-allocation or pricing logic exists
anywhere in this codebase yet (Project.Quotation in ontology/v1.yaml is
reserved, unpopulated) — inventing one here would be fabricating quotation
logic dressed up as infrastructure. calculate_and_record() is the generic
write path a future rule calls through; app.dependency_graph.recompute_dependents
already uses the same changed_by="system_default" write for cascaded
recalculations specifically — this module is for a ONE-OFF calculation not
triggered by a dependency edge (e.g. computed once from values already on
hand, not recomputed every time some upstream node changes)."""

from typing import Any, Optional

from app import graph_store
from app.context_builder import ProposedWrite, apply_to_graph
from app.models import KnowledgeNode


async def calculate_and_record(
    project_id: str, canonical_path: str, node_type: str, value: Any, room_id: Optional[str] = None
) -> KnowledgeNode:
    await apply_to_graph(
        project_id,
        [ProposedWrite(canonical_path=canonical_path, node_type=node_type, value=value, room_id=room_id, tier="optional", changed_by="system_default")],
    )
    return await graph_store.find_one(project_id, canonical_path)


async def list_calculated(project_id: str) -> list[KnowledgeNode]:
    # No value != None filter needed here (unlike app.facts.list_facts):
    # ensure_path's structural containers always default to "user_message",
    # never "system_default" — nothing but a real calculated leaf matches this.
    return await graph_store.find_nodes(project_id, changed_by="system_default")
