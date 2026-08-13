"""Coverage for the active-room-focus + batched-room-question feature
(app.graph._resolve_active_room / app.question_engine.find_knowledge_gaps /
app.graph.generate_question_node / app.graph.build_context_node).

Deliberately does NOT reuse tests/test_graph.py's shared `mock_models`
autouse fixture — that fixture currently fails at setup (patches
app.llm.generate_conflict_confirmation, which doesn't exist on app.llm in
this working tree) due to a separate, already-in-progress conflict-
confirmation refactor that predates this feature and isn't touched by it
(confirmed against HEAD: confirm_conflict_node existed there and is already
gone from the working tree's app/graph.py before any of this file's changes).
This file calls the graph nodes directly with its own minimal, local patches
instead, so it stays runnable independent of that unrelated breakage."""

from unittest.mock import patch

from app.context_builder import ProposedWrite, apply_to_graph
from app.graph import build_context_node, classify_intent_node, generate_question_node
from app.llm import ExtractedFields, GraphExtraction, Operation
from app.tasks import TaskType


def base_state(project_id: str, **overrides) -> dict:
    state = {
        "session_id": "test-session",
        "project_id": project_id,
        "message": "",
        "history": "",
        "skipped_rooms": [],
        "current_field": None,
        "active_room_id": None,
        "intent": [],
        "tasks": [],
        "retrieved": "",
        "pending_question": None,
        "pending_gap": None,
        "pending_operation_questions": None,
        "operation_answers": None,
        "update_summary": None,
        "answer": "",
        "complete": False,
        "needs_answer": False,
        "question_generated": False,
        "wrapup_message": None,
        "trace": [],
    }
    state.update(overrides)
    return state


async def _seed_two_rooms(project_id: str) -> tuple[str, str]:
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.RoomType", node_type="RoomType", value="bedroom", room_id="r2", tier="critical"),
        ],
    )
    return "r1", "r2"


async def test_classify_intent_node_sets_active_room_to_the_last_grounded_write_task():
    project_id = "proj-active-room-last"
    r1, r2 = await _seed_two_rooms(project_id)

    async def fake_classify(message, history="", pending_field=None, *, tree_text="Project", **kwargs):
        return [
            Operation(text="kitchen budget is $10k", intent="CONTEXT_UPDATE", connection=f"Rooms.{r1}"),
            Operation(text="bedroom style is cozy", intent="CONTEXT_UPDATE", connection=f"Rooms.{r2}"),
        ]

    with patch("app.llm.classify_operations", side_effect=fake_classify):
        result = await classify_intent_node(base_state(project_id, message="kitchen budget is $10k, bedroom style is cozy"))

    assert result["active_room_id"] == r2, "the LAST task's grounded room wins, not the first"


async def test_classify_intent_node_leaves_active_room_unchanged_when_nothing_this_turn_grounds_to_a_room():
    project_id = "proj-active-room-unchanged"
    r1, _ = await _seed_two_rooms(project_id)

    async def fake_classify(message, history="", pending_field=None, *, tree_text="Project", **kwargs):
        return [Operation(text=message, intent="DIRECT_ANSWER")]

    with patch("app.llm.classify_operations", side_effect=fake_classify):
        result = await classify_intent_node(base_state(project_id, message="thanks!", active_room_id=r1))

    assert result["active_room_id"] == r1, "a plain DIRECT_ANSWER carries no room signal — prior focus is kept, not cleared"


async def test_generate_question_node_batches_every_open_field_of_the_active_room_into_one_question():
    project_id = "proj-active-room-batch"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
        ],
    )

    captured = {}

    async def fake_generate_question(field_names, context, *, is_retry=False, **kwargs):
        captured["field_names"] = field_names
        return "What's the budget, style, and square footage for the kitchen?"

    with patch("app.llm.generate_question", side_effect=fake_generate_question):
        result = await generate_question_node(base_state(project_id))

    assert captured["field_names"] == ["budget", "style", "square footage", "existing furniture"], (
        "every open room field is passed to the LLM in ONE call, not one call per field"
    )
    assert result["active_room_id"] == "r1"
    assert result["current_field"]["room_id"] == "r1"
    assert len(result["current_field"]["canonical_paths"]) == 4
    assert result["pending_question"] == "What's the budget, style, and square footage for the kitchen?"


async def test_build_context_node_corrects_active_room_id_for_a_brand_new_room():
    project_id = "proj-active-room-new-room"
    # ProjectType is checked before any room field (see
    # find_knowledge_gaps) — seeded here so the room batch this test is
    # actually about isn't masked by that earlier blocking gap.
    await apply_to_graph(
        project_id,
        [ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical")],
    )

    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen", style="modern")

    async def fake_extract_graph_links(message, anchors, recent_nodes, recent_edges, **kwargs):
        return GraphExtraction(), "clean"

    task = None  # single-task path: build_context_node recovers task from state["tasks"]
    from app.tasks import TaskSpec

    state = base_state(
        project_id,
        message="kitchen is modern style",
        tasks=[TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_1", target="kitchen is modern style", connection=None)],
        # A stale room_id left over from an unrelated earlier turn — the new
        # room this write actually creates must override it, not be masked
        # by it (see build_context_node's docstring: connection_room_id or
        # result.room_id is checked BEFORE the old state value).
        active_room_id="some-stale-room-from-a-different-turn",
    )

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await build_context_node(state, task)

    assert result["active_room_id"] not in (None, "some-stale-room-from-a-different-turn")
    assert result["pending_gap"]["room_id"] == result["active_room_id"]
