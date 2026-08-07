from unittest.mock import patch

from app.calculation import calculate_and_record, list_calculated
from app.context_builder import ProposedWrite, apply_to_graph
from app.facts import list_facts
from app.inference import infer_field, list_inferred


async def test_list_facts_returns_only_user_message_nodes():
    project_id = "proj-taxonomy-facts"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical")])
    await calculate_and_record(project_id, "Project.Rooms.r1.Furniture.cabinet.Budget", "Budget", 3000, room_id="r1")

    facts = await list_facts(project_id)
    fact_paths = {n.canonical_path for n in facts}

    assert "Project.Budget.Total" in fact_paths
    assert "Project.Rooms.r1.Furniture.cabinet.Budget" not in fact_paths, "a calculated value must not be counted as a fact"


async def test_infer_field_writes_with_inferred_provenance():
    async def fake_infer(field_label, context, **kwargs):
        return "150 sqft"

    project_id = "proj-taxonomy-inference"
    with patch("app.deepinfra.infer_missing_field", side_effect=fake_infer):
        node = await infer_field(project_id, "Project.Rooms.r1.SquareFootage", "SquareFootage", "square footage", {"roomType": "kitchen"}, room_id="r1")

    assert node.value == "150 sqft"
    assert node.changed_by == "inferred"

    inferred = await list_inferred(project_id)
    assert {n.canonical_path for n in inferred} == {"Project.Rooms.r1.SquareFootage"}


async def test_calculate_and_record_writes_with_system_default_provenance():
    project_id = "proj-taxonomy-calc"
    node = await calculate_and_record(project_id, "Project.Rooms.r1.Furniture.cabinet.Budget", "Budget", 3000, room_id="r1")

    assert node.value == 3000
    assert node.changed_by == "system_default"

    calculated = await list_calculated(project_id)
    assert {n.canonical_path for n in calculated} == {"Project.Rooms.r1.Furniture.cabinet.Budget"}


async def test_all_three_provenances_partition_a_projects_nodes_without_overlap():
    """Exit criterion: 'the quotation output can filter/label by this
    field' — demonstrated generically here (no real quotation feature
    exists yet, see ontology/PHASE12_FACT_INFERENCE_CALCULATION.md): every
    node in a project with a mix of all three provenances is classified
    into exactly one of the three lists, never zero, never more than one."""
    project_id = "proj-taxonomy-partition"

    async def fake_infer(field_label, context, **kwargs):
        return "some inferred value"

    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical")])
    with patch("app.deepinfra.infer_missing_field", side_effect=fake_infer):
        await infer_field(project_id, "Project.Timeline.Value", "Value", "timeline", {})
    await calculate_and_record(project_id, "Project.Budget.Total", "Total", 20000)

    facts = {n.canonical_path for n in await list_facts(project_id)}
    inferred = {n.canonical_path for n in await list_inferred(project_id)}
    calculated = {n.canonical_path for n in await list_calculated(project_id)}

    assert facts == {"Project.BasicInformation.ProjectType"}
    assert inferred == {"Project.Timeline.Value"}
    assert calculated == {"Project.Budget.Total"}
    assert not (facts & inferred) and not (facts & calculated) and not (inferred & calculated)
