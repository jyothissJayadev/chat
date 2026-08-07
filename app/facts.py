"""Facts — Phase 12 (Fact/Inference/Calculation split). A "fact" is a value
written because the client stated it directly, structured or freeform —
changed_by="user_message" (Phase 8/12's provenance taxonomy). This is
already what app.context_builder.apply_to_graph and
app.canonical_mapper._create_instance both default to; this module doesn't
duplicate that write logic, it gives the read side — "which of these values
came directly from the client" — a named, queryable home, per the plan's
three-namespace split. See ontology/PHASE12_FACT_INFERENCE_CALCULATION.md."""

from app.models import KnowledgeNode


async def list_facts(project_id: str) -> list[KnowledgeNode]:
    """Structural container nodes (Project, Project.Rooms, ...) default to
    changed_by="user_message" too — they're not "facts" in any meaningful
    sense, just scaffolding with no value of their own (see
    app.canonical_mapper.ensure_path) — excluded via value != None."""
    nodes = await KnowledgeNode.find(KnowledgeNode.project_id == project_id, KnowledgeNode.changed_by == "user_message").to_list()
    return [n for n in nodes if n.value is not None]
