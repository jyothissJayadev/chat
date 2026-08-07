"""Coverage for multi-clause/multi-room segmentation — Fix 2 of the
deletion-support plan (C:\\Users\\jyoth\\.claude\\plans\\lexical-gliding-clarke.md).

Covers: segment_clauses' boundary/room_hint behavior on a table of real
sentences (including the original transcript's failing multi-room message),
the single-room regression (segmentation must not fire on an ordinary list
within one room), a full understand() -> execution.execute() run proving
writes land under the CORRECT room's subtree, and a direct regression test
for the execute() dispatch bug Fix 2 exposed (Correction 3 — two tasks of the
same type used to silently collapse into one)."""

from unittest.mock import patch

import pytest

from app.deepinfra import ExtractedFields, GraphExtraction, MaterialSpec
from app.execution import execute
from app.models import KnowledgeNode
from app.tasks import TaskSpec, TaskType
from app.understanding import segment_clauses, understand

# ---------------------------------------------------------------------------
# segment_clauses — pure, no DB, table-driven.
# ---------------------------------------------------------------------------

_CASES = [
    (
        "add sofa and couch, tv to living room and let's update the kitchen with a kitchen cabinet and a ceiling fan",
        ["living room", "kitchen"],
    ),
    ("modern style with a chair, sofa set and a tv unit", [None]),
    ("quartz countertops please", [None]),
    ("remove the ceiling fan from the list", [None]),
    ("the kitchen needs a new sink and the bedroom needs a new bed", ["kitchen", "bedroom"]),
    ("$15k for the kitchen, $8k for the bathroom, and $5k for the guest room", ["kitchen", "bathroom", "guest room"]),
]


@pytest.mark.parametrize("message,expected_room_hints", _CASES)
def test_segment_clauses_room_hints(message, expected_room_hints):
    clauses = segment_clauses(message)
    assert [c.room_hint for c in clauses] == expected_room_hints


def test_segment_clauses_does_not_fire_on_a_single_room_item_list():
    """The regression this guards against: an ordinary list of items within
    ONE room ("chair, sofa set and a tv unit") must never be mis-split into
    several fake per-item clauses — segmentation only fires on genuinely
    DIFFERENT rooms, never on "and" alone. Item-level list parsing stays
    extract_fields' own job."""
    clauses = segment_clauses("modern style with a chair, sofa set and a tv unit")
    assert len(clauses) == 1
    assert clauses[0].text == "modern style with a chair, sofa set and a tv unit"


def test_segment_clauses_repeated_mention_of_an_already_seen_room_does_not_reopen_a_boundary():
    clauses = segment_clauses("kitchen needs a sink, and later also fix the kitchen tap, and the bedroom needs paint")
    room_hints = [c.room_hint for c in clauses]
    # Exactly two boundaries (kitchen's FIRST mention, bedroom's) — the
    # second "kitchen" mention folds into the still-open kitchen clause
    # rather than opening a third.
    assert room_hints == ["kitchen", "bedroom"]
    assert "kitchen tap" in clauses[0].text


def test_segment_clauses_trailing_fragment_without_its_own_room_name_inherits_the_preceding_room():
    clauses = segment_clauses("the kitchen needs a sink and the living room needs a rug and a lamp")
    assert [c.room_hint for c in clauses] == ["kitchen", "living room"]
    # "and a lamp" has no room name of its own — it stays part of the
    # still-open living room clause rather than becoming an orphaned clause.
    assert "lamp" in clauses[-1].text


# ---------------------------------------------------------------------------
# understand() -> execution.execute() — writes land under the correct room.
# ---------------------------------------------------------------------------


def base_state(message: str, project_id: str = "proj-seg-e2e") -> dict:
    return {
        "session_id": "test-session", "project_id": project_id, "message": message, "history": "",
        "active_room_id": None, "field_attempts": {}, "intent": [], "tasks": [],
        "retrieved": "", "pending_question": None, "pending_gap": None, "pending_confirmation": None,
        "update_summary": None, "answer": "", "complete": False, "needs_answer": False,
        "question_generated": False, "wrapup_message": None, "trace": [],
    }


async def fake_classify_update_context(message, history="", pending_field=None, **kwargs):
    return ["update_context"]


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
        patch("app.deepinfra.classify_intent", side_effect=fake_classify_update_context),
        patch("app.deepinfra.extract_fields", side_effect=fake_extract_fields_by_room),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        meaning = await understand(message)
        assert len(meaning.tasks) == 2, "one EDIT_CONTEXT task per room clause"

        result = await execute(meaning.tasks, base_state(message, project_id))

    nodes = await KnowledgeNode.find(KnowledgeNode.project_id == project_id).to_list()
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

    async def fake_build_context_node(state, task=None):
        calls.append(task.target)
        return {"update_summary": f"noted {task.target}", "active_room_id": task.room_hint, "trace": []}

    with patch("app.graph.build_context_node", side_effect=fake_build_context_node):
        result = await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="living room facts", room_hint="room-living"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen facts", room_hint="room-kitchen"),
            ],
            base_state("living room facts and kitchen facts"),
        )

    assert calls == ["living room facts", "kitchen facts"], "both tasks must actually be dispatched, not just the first"
    assert "noted living room facts" in result["update_summary"]
    assert "noted kitchen facts" in result["update_summary"]
    # "Active room" after a multi-room turn is whichever room the LAST task
    # touched — see execution.py's own docstring on this design decision.
    assert result["active_room_id"] == "room-kitchen"
