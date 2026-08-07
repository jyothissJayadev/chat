"""Coverage for DELETE_CONTEXT support — see the deletion-support plan
(C:\\Users\\jyoth\\.claude\\plans\\lexical-gliding-clarke.md). Follows the same
direct-async-against-mongomock_motor pattern as tests/test_context_builder.py
and tests/test_graph.py: no new test infrastructure needed.

Covers, in order: a plain moderate-tier retraction and its downstream read-path
effects (materialize_project_summary, find_knowledge_gap), the guard that
tells a removal statement apart from a value edit (the original transcript's
"ceiling fan" regression), the two ways a retraction request can fail to
resolve cleanly (no match, ambiguous match), the confirm_conflict reuse for a
critical-tier removal, room-level cascade deletion, and one full run through
app_graph to prove the routing itself (not just the node function) works."""

from unittest.mock import patch

from app.context_builder import materialize_project_summary
from app.graph import app_graph, confirm_conflict_node, delete_context_node
from app.models import KnowledgeNode
from app.question_engine import find_knowledge_gap
from app.tasks import TaskType
from app.understanding import guard_delete
from app.versioning import retract_node

PROJECT = "proj-delete-1"
ROOM = "room-a"


def _vocab_embed(vocab: dict[str, list[float]], default: list[float] = None):
    """Deterministic fake for app.deepinfra.embed: looks each text up
    (lowercased) in `vocab`, falling back to `default` (or a distinct
    per-call vector) for anything unlisted — same shape as
    test_context_builder.py's fake_embed_routes_to_client_preferences."""
    fallback = default or [0.0, 0.0, 1.0]

    async def fake_embed(texts):
        return [vocab.get(t.strip().lower(), fallback) for t in texts]

    return fake_embed


async def _seed_freeform(project_id: str, room_id: str, node_type: str, label_value: str, canonical_slug: str):
    instance = KnowledgeNode(
        canonical_path=f"Project.Rooms.{room_id}.{node_type}.{canonical_slug}",
        node_type=node_type, project_id=project_id, room_id=room_id,
    )
    await instance.insert()
    label = KnowledgeNode(
        canonical_path=f"{instance.canonical_path}.Label", node_type="Label", parent_id=instance.node_id,
        value=label_value, project_id=project_id, room_id=room_id,
    )
    await label.insert()
    instance.children_ids = [label.node_id]
    await instance.save()
    return instance, label


async def _seed_room(project_id: str, room_id: str, room_type: str):
    room = KnowledgeNode(canonical_path=f"Project.Rooms.{room_id}", node_type="Rooms", project_id=project_id, room_id=room_id)
    await room.insert()
    room_type_node = KnowledgeNode(
        canonical_path=f"Project.Rooms.{room_id}.RoomType", node_type="RoomType", parent_id=room.node_id,
        value=room_type, project_id=project_id, room_id=room_id,
    )
    await room_type_node.insert()
    room.children_ids = [room_type_node.node_id]
    await room.save()
    return room, room_type_node


async def _by_node_id(node_id: str) -> KnowledgeNode:
    # Beanie's Document.get() looks up by the Mongo `_id`/ObjectId, not the
    # business key `node_id` this codebase actually addresses nodes by (see
    # app.models.KnowledgeNode's own docstring on that distinction) — every
    # other module in this codebase re-fetches via find_one on `node_id`
    # instead, so tests do too.
    return await KnowledgeNode.find_one(KnowledgeNode.node_id == node_id)


def base_state(message: str, project_id: str = PROJECT, active_room_id: str | None = None) -> dict:
    return {
        "session_id": "test-session", "project_id": project_id, "message": message, "history": "",
        "active_room_id": active_room_id, "field_attempts": {}, "intent": [], "tasks": [],
        "retrieved": "", "pending_question": None, "pending_gap": None, "pending_confirmation": None,
        "update_summary": None, "answer": "", "complete": False, "needs_answer": False,
        "question_generated": False, "wrapup_message": None, "trace": [],
    }


# ---------------------------------------------------------------------------
# Plain retraction — moderate tier (every freeform node_type today), applies
# immediately, no confirmation needed.
# ---------------------------------------------------------------------------


async def test_delete_context_node_retracts_a_confidently_matched_item():
    project_id = "proj-del-basic"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "ceiling fan", "ceiling_fan")

    with patch("app.canonical_mapper.deepinfra.embed", side_effect=_vocab_embed({})):
        result = await delete_context_node(
            base_state("remove the ceiling fan", project_id=project_id, active_room_id=ROOM)
        )

    refreshed = await _by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"
    assert "ceiling fan" in result["update_summary"]
    assert result.get("pending_confirmation") is None
    assert result.get("pending_question") is None


async def test_retracted_item_is_excluded_from_project_summary():
    project_id = "proj-del-summary"
    instance, label = await _seed_freeform(project_id, ROOM, "Materials", "oak flooring", "oak_flooring")
    await retract_node(instance)

    summary = await materialize_project_summary(project_id)
    # Materials only surface in the summary via the room's own `materials`
    # list, which is keyed off Label/Material/Specification leaves under a
    # LIVE instance path — a retracted instance's Label is simply absent from
    # by_path's active-only fetch, so it can't appear here at all.
    assert summary["rooms"] == []


async def test_retracted_structured_field_reopens_as_a_knowledge_gap():
    project_id = "proj-del-gap"
    # Seed past ProjectType/Budget (checked before any room field) so the
    # walk actually reaches the room-level check this test is about.
    await KnowledgeNode(
        canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", project_id=project_id
    ).insert()
    await KnowledgeNode(canonical_path="Project.Budget.Total", node_type="Total", value="$50k", project_id=project_id).insert()
    room, room_type_node = await _seed_room(project_id, ROOM, "kitchen")

    gap_before = await find_knowledge_gap(project_id, active_room_id=ROOM)
    assert gap_before is not None and gap_before.canonical_path != f"Project.Rooms.{ROOM}.RoomType"

    await retract_node(room_type_node)

    # With the room's RoomType retracted and no other room known,
    # find_knowledge_gap must treat this project as having no room at all —
    # not silently continue past the retracted room as if it were still there.
    gap_after = await find_knowledge_gap(project_id, active_room_id=ROOM)
    assert gap_after is not None
    assert gap_after.node_type == "RoomType"


# ---------------------------------------------------------------------------
# guard_delete — the actual regression: a removal statement must never reach
# build_context/extract_fields as a value edit.
# ---------------------------------------------------------------------------


def test_guard_delete_overrides_edit_context_for_removal_phrasing():
    assert guard_delete(TaskType.EDIT_CONTEXT, "remove the ceiling fan from the list") == TaskType.DELETE_CONTEXT
    assert guard_delete(TaskType.EDIT_CONTEXT, "quartz countertops please") == TaskType.EDIT_CONTEXT
    # Only overrides EDIT_CONTEXT — a genuine question mentioning "remove"
    # isn't retargeted.
    assert guard_delete(TaskType.ANSWER, "what happens if I remove the ceiling fan?") == TaskType.ANSWER


async def test_removal_message_never_touches_a_budget_node():
    project_id = "proj-del-no-budget"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "ceiling fan", "ceiling_fan")

    with patch("app.canonical_mapper.deepinfra.embed", side_effect=_vocab_embed({})):
        await delete_context_node(
            base_state("remove the ceiling fan from the list", project_id=project_id, active_room_id=ROOM)
        )

    budget_nodes = await KnowledgeNode.find(KnowledgeNode.project_id == project_id, KnowledgeNode.node_type == "Budget").to_list()
    assert budget_nodes == []


# ---------------------------------------------------------------------------
# Failure-to-resolve paths — never guess.
# ---------------------------------------------------------------------------


async def test_no_confident_match_asks_for_clarification_instead_of_retracting():
    project_id = "proj-del-nomatch"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "sofa", "sofa")

    with patch("app.canonical_mapper.deepinfra.embed", side_effect=_vocab_embed({"pergola": [1.0, 0.0, 0.0], "sofa": [0.0, 1.0, 0.0]})):
        result = await delete_context_node(base_state("remove the pergola", project_id=project_id, active_room_id=ROOM))

    assert result.get("pending_question") is not None
    assert "pergola" in result["pending_question"]
    refreshed = await _by_node_id(instance.node_id)
    assert refreshed.lifecycle == "active"


async def test_ambiguous_match_lists_candidates_instead_of_guessing():
    project_id = "proj-del-ambiguous"
    room_b = "room-b"
    inst_a, label_a = await _seed_freeform(project_id, ROOM, "Furniture", "ceiling fan", "ceiling_fan")
    inst_b, label_b = await _seed_freeform(project_id, room_b, "Furniture", "ceiling fan", "ceiling_fan")

    vocab = {"the fan": [1.0, 0.0, 0.0], "ceiling fan": [1.0, 0.0, 0.0]}
    with patch("app.canonical_mapper.deepinfra.embed", side_effect=_vocab_embed(vocab)):
        # No active_room_id / room_hint — both rooms' "ceiling fan" are
        # equally valid candidates, so this must not silently pick one.
        result = await delete_context_node(base_state("remove the fan", project_id=project_id))

    assert result.get("pending_question") is not None
    assert result.get("pending_confirmation") is None
    for inst in (inst_a, inst_b):
        refreshed = await _by_node_id(inst.node_id)
        assert refreshed.lifecycle == "active"


# ---------------------------------------------------------------------------
# confirm_conflict_node — reused, not duplicated, for a delete's "kind".
# ---------------------------------------------------------------------------


async def test_confirm_conflict_node_retracts_on_affirmative_delete_reply():
    project_id = "proj-del-confirm-yes"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "console table", "console_table")

    state = base_state("yes", project_id=project_id)
    state["pending_confirmation"] = {
        "kind": "delete", "canonical_path": instance.canonical_path, "node_type": instance.node_type,
        "old_value": "console table", "new_value": None, "tier": "critical", "question": "Remove the console table?",
    }
    await confirm_conflict_node(state)

    refreshed = await _by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"


async def test_confirm_conflict_node_keeps_node_on_negative_delete_reply():
    project_id = "proj-del-confirm-no"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "console table", "console_table")

    state = base_state("no, keep it", project_id=project_id)
    state["pending_confirmation"] = {
        "kind": "delete", "canonical_path": instance.canonical_path, "node_type": instance.node_type,
        "old_value": "console table", "new_value": None, "tier": "critical", "question": "Remove the console table?",
    }
    await confirm_conflict_node(state)

    refreshed = await _by_node_id(instance.node_id)
    assert refreshed.lifecycle == "active"


# ---------------------------------------------------------------------------
# Room-level cascade deletion — always held for confirmation, always
# retract_subtree on affirmative.
# ---------------------------------------------------------------------------


async def test_room_deletion_is_held_for_confirmation():
    project_id = "proj-del-room"
    room, room_type_node = await _seed_room(project_id, ROOM, "living room")
    material_instance, material_label = await _seed_freeform(project_id, ROOM, "Materials", "oak flooring", "oak_flooring")
    material_instance.parent_id = room.node_id
    await material_instance.save()

    with patch("app.deepinfra.generate_conflict_confirmation_delete", side_effect=lambda old_value, **kw: f"Remove {old_value}?"):
        result = await delete_context_node(base_state("remove the living room from the project", project_id=project_id))

    assert result.get("pending_confirmation") is not None
    assert result["pending_confirmation"]["kind"] == "delete_room"
    # Nothing retracted yet — only confirmed.
    assert (await _by_node_id(room.node_id)).lifecycle == "active"
    assert (await _by_node_id(room_type_node.node_id)).lifecycle == "active"


async def test_room_deletion_cascades_to_descendants_on_confirmation():
    project_id = "proj-del-room-cascade"
    room, room_type_node = await _seed_room(project_id, ROOM, "living room")
    material_instance, material_label = await _seed_freeform(project_id, ROOM, "Materials", "oak flooring", "oak_flooring")
    material_instance.parent_id = room.node_id
    room.children_ids = [room_type_node.node_id, material_instance.node_id]
    await room.save()

    state = base_state("yes", project_id=project_id)
    state["pending_confirmation"] = {
        "kind": "delete_room", "canonical_path": room.canonical_path, "node_type": "Rooms",
        "old_value": "living room", "new_value": None, "tier": "critical", "question": "Remove living room?",
    }
    await confirm_conflict_node(state)

    for node_id in (room.node_id, room_type_node.node_id, material_instance.node_id, material_label.node_id):
        refreshed = await _by_node_id(node_id)
        assert refreshed.lifecycle == "retracted", refreshed.canonical_path


# ---------------------------------------------------------------------------
# End-to-end: the routing itself (not just the node function) reaches
# delete_context for a single-task removal turn.
# ---------------------------------------------------------------------------


async def fake_classify_delete(message, history="", pending_field=None, **kwargs):
    return ["update_context"]


async def test_end_to_end_routing_reaches_delete_context():
    project_id = "proj-del-e2e"
    instance, label = await _seed_freeform(project_id, ROOM, "Furniture", "ceiling fan", "ceiling_fan")

    state = base_state("remove the ceiling fan", project_id=project_id, active_room_id=ROOM)
    with (
        patch("app.deepinfra.classify_intent", side_effect=fake_classify_delete),
        patch("app.canonical_mapper.deepinfra.embed", side_effect=_vocab_embed({})),
    ):
        final_state = None
        async for mode, chunk in app_graph.astream(state, stream_mode=["values"]):
            final_state = chunk

    node_names = [t.node_name for t in final_state["trace"]]
    assert "delete_context" in node_names
    assert "build_context" not in node_names
    refreshed = await _by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"
