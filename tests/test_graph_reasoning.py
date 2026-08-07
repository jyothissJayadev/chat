from app.context_builder import ProposedWrite, apply_to_graph
from app.dependency_graph import add_edge
from app.graph_reasoning import items_depending_on, rooms_exceeding_budget, rooms_with_material
from app.models import KnowledgeNode


async def _room_line_item(project_id: str, room_id: str, item_slug: str, amount) -> None:
    """Seeds a Project.Quotation.RoomLineItems.<instance>.Amount leaf tagged
    to room_id via KnowledgeNode.room_id metadata — see
    app/graph_reasoning.py's docstring for why it's not path-nested under
    the room. Nothing in the app writes these yet, so tests seed them
    directly, same as app.dependency_graph's own worked-example tests."""
    path = f"Project.Quotation.RoomLineItems.{item_slug}.Amount"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path=path, node_type="Amount", value=amount, room_id=room_id, tier="optional")])
    node = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == path)
    node.room_id = room_id
    await node.save()


async def test_rooms_exceeding_budget_finds_the_one_room_that_actually_exceeds():
    project_id = "proj-reasoning-budget"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.kitchen1.RoomType", node_type="RoomType", value="kitchen", room_id="kitchen1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.kitchen1.Budget", node_type="Budget", value="$8k", room_id="kitchen1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.bedroom1.RoomType", node_type="RoomType", value="bedroom", room_id="bedroom1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.bedroom1.Budget", node_type="Budget", value="$5k", room_id="bedroom1", tier="critical"),
        ],
    )
    await _room_line_item(project_id, "kitchen1", "cabinets", "$6k")
    await _room_line_item(project_id, "kitchen1", "flooring", "$3.5k")  # kitchen total: $9.5k > $8k budget
    await _room_line_item(project_id, "bedroom1", "bed", "$2k")  # bedroom total: $2k < $5k budget

    overages = await rooms_exceeding_budget(project_id)

    assert len(overages) == 1
    assert overages[0].room_id == "kitchen1"
    assert overages[0].room_type == "kitchen"
    assert overages[0].budget == 8000.0
    assert overages[0].spent == 9500.0
    assert overages[0].overage == 1500.0


async def test_rooms_exceeding_budget_skips_rooms_with_no_line_items_or_unparseable_budget():
    project_id = "proj-reasoning-budget-skip"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="office", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="whatever works", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.RoomType", node_type="RoomType", value="den", room_id="r2", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.Budget", node_type="Budget", value="$1k", room_id="r2", tier="critical"),
        ],
    )
    await _room_line_item(project_id, "r1", "desk", "$5k")  # budget doesn't parse -> skipped

    assert await rooms_exceeding_budget(project_id) == []  # r1 skipped (unparseable budget), r2 has no line items


async def test_rooms_with_material_matches_synonymous_material_names():
    project_id = "proj-reasoning-material"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.kitchen1.RoomType", node_type="RoomType", value="kitchen", room_id="kitchen1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.kitchen1.Materials.flooring.Material", node_type="Material", value="oak wood flooring", room_id="kitchen1", tier="optional"),
            ProposedWrite(canonical_path="Project.Rooms.bedroom1.RoomType", node_type="RoomType", value="bedroom", room_id="bedroom1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.bedroom1.Materials.flooring.Material", node_type="Material", value="quartz countertop", room_id="bedroom1", tier="optional"),
        ],
    )

    matches = await rooms_with_material(project_id, "oak wood")

    assert len(matches) == 1
    assert matches[0].room_id == "kitchen1"
    assert matches[0].room_type == "kitchen"


async def test_rooms_with_material_returns_one_ref_per_room_not_per_matching_leaf():
    project_id = "proj-reasoning-material-dedup"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.k1.RoomType", node_type="RoomType", value="kitchen", room_id="k1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.k1.Materials.flooring.Material", node_type="Material", value="oak wood", room_id="k1", tier="optional"),
            ProposedWrite(canonical_path="Project.Rooms.k1.Materials.cabinet.Material", node_type="Material", value="oak wood", room_id="k1", tier="optional"),
        ],
    )

    matches = await rooms_with_material(project_id, "oak wood")
    assert len(matches) == 1


async def test_items_depending_on_delegates_to_the_dependency_graph():
    project_id = "proj-reasoning-dependents"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.k1.Materials.cabinet.Material", node_type="Material", value="oak wood", room_id="k1", tier="optional"),
            ProposedWrite(canonical_path="Project.Quotation.RoomLineItems.cabinet.Amount", node_type="Amount", value=1200, room_id="k1", tier="optional"),
        ],
    )
    material = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == "Project.Rooms.k1.Materials.cabinet.Material")
    line_item = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == "Project.Quotation.RoomLineItems.cabinet.Amount")
    await add_edge(line_item.node_id, material.node_id, "derives_from", project_id)

    dependents = await items_depending_on(project_id, material.node_id)

    assert len(dependents) == 1
    assert dependents[0].node_id == line_item.node_id
