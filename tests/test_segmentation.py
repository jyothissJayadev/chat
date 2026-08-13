"""Coverage for multi-operation turns spanning more than one room — originally
written for Fix 2 of the deletion-support plan
(C:\\Users\\jyoth\\.claude\\plans\\lexical-gliding-clarke.md), which added a
regex room-splitter (segment_clauses) ahead of the single-label classifier of
the time. That splitter is gone — app.llm.classify_operations now does
its own multi-operation splitting directly (see
tests/test_understanding.py's golden set for its own room-split coverage) —
but the behaviors this file guards are still real: a full understand() ->
execution.execute() run proving writes land under the CORRECT room's
subtree, and a direct regression test for the execute() dispatch bug Fix 2
originally exposed (Correction 3 — two tasks of the same type used to
silently collapse into one)."""

from unittest.mock import patch

from app import graph_store
from app.llm import ExtractedFields, GraphExtraction, MaterialSpec
from app.execution import execute
from app.tasks import TaskSpec, TaskType
from app.understanding import understand

# ---------------------------------------------------------------------------
# understand() -> execution.execute() — writes land under the correct room.
# ---------------------------------------------------------------------------


def base_state(message: str, project_id: str = "proj-seg-e2e") -> dict:
    return {
        "session_id": "test-session", "project_id": project_id, "message": message, "history": "",
        "skipped_rooms": [], "current_field": None, "intent": [], "tasks": [],
        "retrieved": "", "pending_question": None, "pending_gap": None, "pending_confirmation": None,
        "update_summary": None, "answer": "", "complete": False, "needs_answer": False,
        "question_generated": False, "wrapup_message": None, "trace": [],
    }


async def fake_classify_operations_by_room(message, history="", pending_field=None, **kwargs):
    """Stands in for app.llm.classify_operations in the e2e test below —
    splits a two-room message into two CONTEXT_UPDATE operations the same way
    the real classifier's RULE 1/RULE 11 would, so the test can exercise
    understand() -> execute() without a live LLM call."""
    from app.llm import Operation

    return [
        Operation(text="add sofa and couch to living room", intent="CONTEXT_UPDATE"),
        Operation(
            text="update the kitchen with a kitchen cabinet and a ceiling fan",
            intent="CONTEXT_UPDATE",
        ),
    ]


async def fake_extract_fields_by_room(message, known, **kwargs):
    lower = message.lower()
    if "kitchen" in lower:
        return ExtractedFields(
            roomType="kitchen",
            materials=[MaterialSpec(item="kitchen cabinet", material="oak"), MaterialSpec(item="ceiling fan", material="steel")],
        )
    return ExtractedFields(
        roomType="living room",
        materials=[MaterialSpec(item="sofa", material="fabric"), MaterialSpec(item="couch", material="fabric")],
    )


async def fake_extract_graph_links(message, anchors, candidate_nodes, recent_edges, **kwargs):
    return GraphExtraction(), "clean"


async def test_multi_room_message_writes_each_rooms_facts_under_its_own_subtree():
    project_id = "proj-seg-multiroom"
    message = "add sofa and couch to living room and let's update the kitchen with a kitchen cabinet and a ceiling fan"

    with (
        patch("app.llm.classify_operations", side_effect=fake_classify_operations_by_room),
        patch("app.llm.extract_fields", side_effect=fake_extract_fields_by_room),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        meaning = await understand(message)
        assert len(meaning.tasks) == 2, "one EDIT_CONTEXT task per room operation"

        result = await execute(meaning.tasks, base_state(message, project_id))

    nodes = await graph_store.find_nodes(project_id)
    living_room_id = next(n.room_id for n in nodes if n.node_type == "RoomType" and n.value == "living room")
    kitchen_id = next(n.room_id for n in nodes if n.node_type == "RoomType" and n.value == "kitchen")
    assert living_room_id != kitchen_id

    living_labels = {n.value for n in nodes if n.room_id == living_room_id and n.node_type == "Label"}
    kitchen_labels = {n.value for n in nodes if n.room_id == kitchen_id and n.node_type == "Label"}
    assert living_labels == {"sofa", "couch"}
    assert kitchen_labels == {"kitchen cabinet", "ceiling fan"}
    # Neither room's facts leaked into the other's subtree.
    assert "kitchen cabinet" not in living_labels
    assert "sofa" not in kitchen_labels
    assert result["needs_answer"] is True


# ---------------------------------------------------------------------------
# execute() — direct regression test for Correction 3: two tasks of the same
# TaskType must both be dispatched and both results merged, not silently
# collapsed into one (the bug a type-keyed dispatch table had before this fix).
# ---------------------------------------------------------------------------


async def test_execute_dispatches_two_same_type_tasks_independently():
    calls: list[str] = []

    async def fake_build_context_node(state, task=None, resolved=None):
        calls.append(task.target)
        return {
            "update_summary": f"noted {task.target}",
            "pending_gap": {
                "canonical_path": f"Project.Rooms.{task.room_hint}.Style", "field_label": "style",
                "node_type": "Style", "room_id": task.room_hint,
            },
            "trace": [],
        }

    async def fake_extract_empty(message, known, **kwargs):
        return ExtractedFields()

    # Distinct, unmatched room_hints (neither "room-living" nor "room-kitchen"
    # is a real room name any existing room could fuzzy-match) so the two
    # tasks resolve to two DIFFERENT new rooms and land in separate clusters
    # — this is what proves they're still dispatched independently under the
    # resolve+cluster+commit pipeline, not just "both eventually called".
    with (
        patch("app.graph.build_context_node", side_effect=fake_build_context_node),
        patch("app.llm.extract_fields", side_effect=fake_extract_empty),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="living room facts", room_hint="room-living"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen facts", room_hint="room-kitchen"),
            ],
            base_state("living room facts and kitchen facts"),
        )

    assert sorted(calls) == ["kitchen facts", "living room facts"], "both tasks must actually be dispatched, not just the first"
    assert "noted living room facts" in result["update_summary"]
    assert "noted kitchen facts" in result["update_summary"]
    # pending_gap merges "first non-null wins" (mirrors pending_confirmation)
    # — the first task's own sub-result survives into the merged state.
    assert result["pending_gap"]["room_id"] == "room-living"
