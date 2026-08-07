"""Inference — Phase 12 (Fact/Inference/Calculation split). Relocates
app.deepinfra.infer_missing_field's ROLE — not a copy of the function
itself, deepinfra.py stays the one place LLM calls are made — into a named,
changed_by="inferred" write path.

This gives app.context_builder's pipeline the terminal fallback
save_project_node already has today (a field nobody ever answers gets a
reasonable, LLM-grounded guess instead of staying blank forever) — Phase 7
explicitly didn't port this (see ontology/PHASE7_CONTEXT_BUILDER.md); this
phase does. Not called from app.context_builder.build_context yet — nothing
there decides "the client is done answering, fill in what's left" (that
decision belongs to completeness logic, which is Phase 15's job — see
ontology/PHASE15_KNOWLEDGE_GAP.md). infer_field is a real, tested
capability waiting for that caller."""

from typing import Any, Optional

from app import deepinfra
from app.context_builder import ProposedWrite, apply_to_graph
from app.models import KnowledgeNode


async def infer_field(
    project_id: str,
    canonical_path: str,
    node_type: str,
    field_label: str,
    context: dict[str, Any],
    room_id: Optional[str] = None,
) -> KnowledgeNode:
    """field_label is what's shown to the model (e.g. "square footage"),
    distinct from canonical_path/node_type (where the answer is stored) —
    same split infer_missing_field's own (field_name, context) signature
    already has today."""
    value = await deepinfra.infer_missing_field(field_label, context)
    await apply_to_graph(
        project_id,
        [ProposedWrite(canonical_path=canonical_path, node_type=node_type, value=value, room_id=room_id, tier="optional", changed_by="inferred")],
    )
    return await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == canonical_path)


async def list_inferred(project_id: str) -> list[KnowledgeNode]:
    # No value != None filter needed here (unlike app.facts.list_facts):
    # ensure_path's structural containers always default to "user_message",
    # never "inferred" — nothing but a real inferred leaf can match this query.
    return await KnowledgeNode.find(KnowledgeNode.project_id == project_id, KnowledgeNode.changed_by == "inferred").to_list()
