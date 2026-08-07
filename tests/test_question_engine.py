from unittest.mock import patch

from app.context_builder import ProposedWrite, apply_to_graph
from app.question_engine import find_knowledge_gap, generate_question


async def test_find_knowledge_gap_starts_with_project_type_on_an_empty_project():
    gap = await find_knowledge_gap("proj-gap-empty")
    assert gap.canonical_path == "Project.BasicInformation.ProjectType"
    assert gap.tier == "critical"


async def test_find_knowledge_gap_moves_to_budget_once_project_type_is_set():
    project_id = "proj-gap-budget"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical")])

    gap = await find_knowledge_gap(project_id)
    assert gap.canonical_path == "Project.Budget.Total"


async def test_find_knowledge_gap_asks_for_a_new_room_when_project_fields_are_done_but_no_room_exists():
    project_id = "proj-gap-no-room"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
        ],
    )

    gap = await find_knowledge_gap(project_id)
    assert gap.canonical_path == "Project.Rooms.<new>"
    assert gap.node_type == "RoomType"


async def test_find_knowledge_gap_walks_room_fields_in_priority_order():
    project_id = "proj-gap-room-fields"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
        ],
    )

    gap = await find_knowledge_gap(project_id, active_room_id="r1")
    assert gap.canonical_path == "Project.Rooms.r1.Budget"

    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical")])
    gap = await find_knowledge_gap(project_id, active_room_id="r1")
    assert gap.canonical_path == "Project.Rooms.r1.Style"


async def test_find_knowledge_gap_reaches_timeline_after_every_room_field_is_resolved():
    project_id = "proj-gap-timeline"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
        ],
    )

    gap = await find_knowledge_gap(project_id, active_room_id="r1")
    assert gap.canonical_path == "Project.Timeline.Value"


async def test_find_knowledge_gap_returns_none_once_everything_is_resolved():
    project_id = "proj-gap-complete"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
            ProposedWrite(canonical_path="Project.Timeline.Value", node_type="Value", value="6 weeks", tier="moderate"),
        ],
    )

    assert await find_knowledge_gap(project_id, active_room_id="r1") is None


async def test_generate_question_passes_the_gaps_field_label_to_deepinfra():
    captured = {}

    async def fake_generate_question(field_name, context, *, is_retry=False, capture=None):
        captured["field_name"] = field_name
        captured["is_retry"] = is_retry
        return "What's the overall budget for this project?"

    gap = (await find_knowledge_gap("proj-gap-question-text"))
    with patch("app.deepinfra.generate_question", side_effect=fake_generate_question):
        question = await generate_question(gap, {}, is_retry=True)

    assert question == "What's the overall budget for this project?"
    assert captured["field_name"] == "project type"
    assert captured["is_retry"] is True
