from unittest.mock import patch

from app import graph_store
from app.llm import ExtractedFields, GraphExtraction
from app.execution import execute, merge_task_results
from app.models import KnowledgeNode
from app.tasks import TaskSpec, TaskType


def base_state(message: str, project_id: str = "proj-exec-1") -> dict:
    return {
        "session_id": "test-session",
        "project_id": project_id,
        "message": message,
        "history": "",
        "skipped_rooms": [],
        "current_field": None,
        "intent": [],
        "tasks": [],
        "retrieved": "",
        "pending_question": None,
        "pending_gap": None,
        "pending_confirmation": None,
        "update_summary": None,
        "answer": "",
        "complete": False,
        "needs_answer": False,
        "question_generated": False,
        "wrapup_message": None,
        "trace": [],
    }


async def fake_extract(message, known, **kwargs):
    return ExtractedFields(roomType="kitchen", style="modern")


async def fake_extract_graph_links(message, anchors, recent_nodes, recent_edges, **kwargs):
    return GraphExtraction(), "clean"


async def fake_query_catalog(query, limit=5, style_tags=None):
    return []


async def test_execute_single_edit_context_task_writes_via_build_context():
    """A single-EDIT_CONTEXT call never actually reaches execute() in the live
    graph (that routes straight to the build_context node — see
    app.graph._route_intent), but execute() itself always flags needs_answer
    whenever EDIT_CONTEXT is in the batch, regardless of batch size — it has
    no way to know from the task list alone whether the caller is really a
    split turn, and app.graph.handle_split_intents_node (the only real
    caller) is only ever invoked for genuine multi-task batches."""
    project_id = "proj-exec-single-edit"
    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("modern kitchen", project_id))

    assert result["needs_answer"] is True
    assert "style: modern" in result["update_summary"]
    nodes = await graph_store.find_nodes(project_id)
    room_id = next(n.room_id for n in nodes if n.node_type == "RoomType" and n.value == "kitchen")
    style_node = await graph_store.find_one(project_id, f"Project.Rooms.{room_id}.Style")
    assert style_node.value == "modern"


async def test_execute_single_delete_context_task_retracts_via_delete_context_node():
    project_id = "proj-exec-delete"
    # delete_context_node's matching only searches the five freeform types
    # (see app.canonical_mapper._FREEFORM_NODE_TYPES), not structured slot
    # leaves — a Furniture instance + Label, matching what
    # canonical_mapper._create_instance actually produces.
    instance = await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.Furniture.sofa", node_type="Furniture", project_id=project_id, room_id="r1")
    )
    await graph_store.insert_node(
        KnowledgeNode(
            canonical_path="Project.Rooms.r1.Furniture.sofa.Label", node_type="Label", parent_id=instance.node_id,
            value="sofa", project_id=project_id, room_id="r1",
        )
    )

    async def fake_embed(texts):
        # "remove the sofa" (the task's own target text) and "sofa" (the
        # existing Label's value) both need a real, matching non-zero
        # vector — a zero vector's cosine similarity is defined as 0.0 by
        # app.canonical_mapper._cosine_similarity's own guard, which would
        # read as "no match" here rather than a confident one.
        return [[1.0, 0.0, 0.0] for _ in texts]

    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        result = await execute(
            [TaskSpec(type=TaskType.DELETE_CONTEXT, target="remove the sofa", room_hint="r1")],
            base_state("remove the sofa", project_id),
        )

    assert result["needs_answer"] is True
    assert "sofa" in result["update_summary"]
    refreshed = await graph_store.find_by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"


async def test_execute_single_database_query_task_returns_retrieved():
    with patch("app.rag.query_catalog", side_effect=fake_query_catalog):
        result = await execute([TaskSpec(type=TaskType.DATABASE_QUERY)], base_state("find a sofa"))

    assert result["retrieved"] == "no matches"
    assert "update_summary" not in result


async def test_execute_retrieve_context_task_summarizes_known_fields():
    project_id = "proj-exec-retrieve"
    with (
        patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("budget is $20k", project_id))

    result = await execute([TaskSpec(type=TaskType.RETRIEVE_CONTEXT)], base_state("what's my budget", project_id))
    assert "overallBudget=$20k" in result["retrieved"]


async def test_execute_answer_only_dispatches_nothing_and_does_not_flag_needs_answer():
    """An ANSWER-only batch never actually reaches execute() in the live graph
    (single-task ANSWER routes straight to generate_answer_node), but if it
    did, there's nothing to dispatch and no EDIT_CONTEXT to validate/complete
    afterward — needs_answer is EDIT_CONTEXT-gated (see execution.py's
    docstring), not ANSWER-gated."""
    result = await execute([TaskSpec(type=TaskType.ANSWER)], base_state("hello"))
    assert result == {"trace": []}


async def test_execute_edit_context_plus_answer_writes_and_flags_answer():
    project_id = "proj-exec-split-answer"
    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT), TaskSpec(type=TaskType.ANSWER)],
            base_state("modern kitchen, and what would oak flooring cost", project_id),
        )

    assert result["needs_answer"] is True
    nodes = await graph_store.find_nodes(project_id)
    room_id = next(n.room_id for n in nodes if n.node_type == "RoomType" and n.value == "kitchen")
    style_node = await graph_store.find_one(project_id, f"Project.Rooms.{room_id}.Style")
    assert style_node.value == "modern"


async def test_execute_database_query_plus_retrieve_context_merges_both_retrievals():
    project_id = "proj-exec-dq-rc"
    with (
        patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("budget is $20k", project_id))

    with patch("app.rag.query_catalog", side_effect=fake_query_catalog):
        result = await execute(
            [TaskSpec(type=TaskType.DATABASE_QUERY), TaskSpec(type=TaskType.RETRIEVE_CONTEXT)], base_state("remind me and find a sofa", project_id)
        )

    assert "no matches" in result["retrieved"]
    assert "overallBudget=$20k" in result["retrieved"]
    assert "update_summary" not in result
    assert "needs_answer" not in result


async def test_execute_edit_context_conflict_sets_pending_confirmation_and_needs_answer():
    project_id = "proj-exec-conflict"
    with (
        patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$15k")),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("budget is $15k", project_id))

    async def fake_confirmation(field_label, old_value, new_value, **kwargs):
        return f"You said {old_value} before — did you mean to change it to {new_value}?"

    with (
        patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.llm.generate_conflict_confirmation", side_effect=fake_confirmation),
    ):
        result = await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT), TaskSpec(type=TaskType.ANSWER)],
            base_state("actually the budget is $20k, and what would flooring cost", project_id),
        )

    assert result["pending_confirmation"]["canonical_path"] == "Project.Budget.Total"
    assert result["pending_confirmation"]["old_value"] == "$15k"
    assert result["pending_confirmation"]["new_value"] == "$20k"
    assert result["question_generated"] is True
    assert result["needs_answer"] is True


# ---------------------------------------------------------------------------
# Write clustering — Step 3 of the classifier-redesign plan (see
# memory/classifier_redesign_decisions.md): connected EDIT_CONTEXT/
# DELETE_CONTEXT tasks (same resolved target) commit sequentially, delete
# before edit, so one turn's operations can't race each other into a
# duplicate room/entity or an out-of-order delete-then-recreate. Unconnected
# tasks still run independently (already covered by
# test_execute_dispatches_two_same_type_tasks_independently in
# tests/test_segmentation.py).
# ---------------------------------------------------------------------------


async def test_execute_two_operations_creating_the_same_new_room_do_not_duplicate_it():
    """The core race this clustering exists to prevent: two operations in one
    turn both resolve (independently, in parallel) against a room that
    doesn't exist yet — without clustering, each would allocate its own new
    room id and both would commit, creating two "kitchen" rooms instead of
    one."""
    project_id = "proj-exec-no-duplicate-room"

    async def fake_extract_dispatch(message, known, **kwargs):
        # Branches on the task's own message text rather than call order —
        # resolve_context runs for both tasks concurrently (see
        # execution._resolve_write_task), so which one's extract_fields call
        # actually lands first isn't guaranteed.
        if "budget" in message:
            return ExtractedFields(roomType="kitchen", budgetOrRequirement="$8k")
        return ExtractedFields(roomType="kitchen", style="modern")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_dispatch),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen should be modern"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen budget is $8k"),
            ],
            base_state("kitchen should be modern, kitchen budget is $8k", project_id),
        )

    rooms = [n for n in await graph_store.find_nodes(project_id) if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms"]
    assert len(rooms) == 1, f"expected exactly one kitchen room, got {len(rooms)}"
    style_node = await graph_store.find_one(project_id, f"Project.Rooms.{rooms[0].room_id}.Style")
    budget_node = await graph_store.find_one(project_id, f"Project.Rooms.{rooms[0].room_id}.Budget")
    assert style_node.value == "modern", "both operations' facts must land on the SAME room"
    assert budget_node.value == "$8k"
    assert result["needs_answer"] is True


async def test_execute_delete_commits_before_edit_within_the_same_connected_cluster():
    project_id = "proj-exec-delete-before-edit"
    instance = await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.Furniture.sofa", node_type="Furniture", project_id=project_id, room_id="r1")
    )
    await graph_store.insert_node(
        KnowledgeNode(
            canonical_path="Project.Rooms.r1.Furniture.sofa.Label", node_type="Label", parent_id=instance.node_id,
            value="sofa", project_id=project_id, room_id="r1",
        )
    )
    await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", project_id=project_id, room_id="r1")
    )

    order: list[str] = []

    import app.context_builder as context_builder_module
    import app.versioning as versioning_module

    _real_retract_node = versioning_module.retract_node
    _real_apply_to_graph = context_builder_module.apply_to_graph

    async def tracking_retract_node(node, *args, **kwargs):
        order.append("delete")
        return await _real_retract_node(node, *args, **kwargs)

    async def tracking_apply_to_graph(project_id_, writes):
        if writes:
            order.append("edit")
        return await _real_apply_to_graph(project_id_, writes)

    async def fake_extract_kitchen_style(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen", style="modern")

    async def fake_embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    with (
        patch("app.versioning.retract_node", side_effect=tracking_retract_node),
        patch("app.context_builder.apply_to_graph", side_effect=tracking_apply_to_graph),
        patch("app.llm.extract_fields", side_effect=fake_extract_kitchen_style),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed),
    ):
        result = await execute(
            [
                TaskSpec(type=TaskType.DELETE_CONTEXT, target="remove the sofa", room_hint="kitchen"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen style is modern", room_hint="kitchen"),
            ],
            base_state("remove the sofa, kitchen style is modern", project_id),
        )

    assert order == ["delete", "edit"], f"delete must commit before edit in a connected cluster, got {order}"
    refreshed = await graph_store.find_by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"
    style_node = await graph_store.find_one(project_id, "Project.Rooms.r1.Style")
    assert style_node.value == "modern"
    assert result["needs_answer"] is True


async def test_execute_unconnected_operations_each_get_their_own_room():
    """Two operations naming DIFFERENT rooms must not be forced into the same
    cluster — each resolves and commits to its own room, independently."""
    project_id = "proj-exec-unconnected-rooms"

    async def fake_extract_dispatch(message, known, **kwargs):
        if "bedroom" in message:
            return ExtractedFields(roomType="bedroom", style="cozy")
        return ExtractedFields(roomType="living room", style="modern")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_dispatch),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="living room is modern"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="bedroom is cozy"),
            ],
            base_state("living room is modern, bedroom is cozy", project_id),
        )

    rooms = [n for n in await graph_store.find_nodes(project_id) if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms"]
    assert len(rooms) == 2, "two distinct rooms named in the turn must not be merged into one"


# ---------------------------------------------------------------------------
# Connection-aware resolve & cluster (classifier-connection plan, Phase 3)
# ---------------------------------------------------------------------------


async def test_execute_connection_grounds_room_even_when_extraction_names_no_room():
    """A grounded connection (e.g. "Rooms.r1") is a task's own room-grounding
    for its resolve_context call — even when extraction itself doesn't name
    a room, the connection alone is enough to land the write on the right
    existing room instead of minting a stray new one (there's no
    active_room_id fallback anymore — removed system-wide)."""
    project_id = "proj-exec-connection-room-override"
    await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1", node_type="Rooms", project_id=project_id, room_id="r1")
    )
    await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", project_id=project_id, room_id="r1")
    )

    async def fake_extract_no_room(message, known, **kwargs):
        return ExtractedFields(style="modern")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_no_room),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT, target="make it modern", connection="Rooms.r1")],
            base_state("make it modern", project_id),
        )

    style_node = await graph_store.find_one(project_id, "Project.Rooms.r1.Style")
    assert style_node is not None and style_node.value == "modern"
    rooms = [n for n in await graph_store.find_nodes(project_id) if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms"]
    assert len(rooms) == 1, "connection must land the write on the existing room, not mint a new one"


async def test_execute_free_text_connection_clusters_new_room_creation():
    """A free-text (not-yet-existing) connection shared by several
    operations overrides room_hint for each — same as a static
    regex-detected room_hint would, just sourced from the classifier's
    project-grounded guess. Two operations sharing one still create exactly
    ONE new room, via the existing new-room: clustering key."""
    project_id = "proj-exec-connection-new-room"

    async def fake_extract_empty(message, known, **kwargs):
        return ExtractedFields()

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_empty),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="the living room should be modern", connection="Living Room"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="add a sofa to the living room", connection="Living Room"),
            ],
            base_state("the living room should be modern, add a sofa to the living room", project_id),
        )

    rooms = [n for n in await graph_store.find_nodes(project_id) if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms"]
    assert len(rooms) == 1, f"expected exactly one new room, got {len(rooms)}"
    room_type_node = await graph_store.find_one(project_id, f"Project.Rooms.{rooms[0].room_id}.RoomType")
    assert room_type_node.value == "Living Room"


async def test_execute_delete_connection_room_override_matches_edit_for_clustering():
    """A DELETE_CONTEXT task's connection room override (via
    app.graph.resolve_delete_target) must key-match an EDIT_CONTEXT task's
    own connection room override for the SAME room, so the two still cluster
    (and delete still commits before edit) exactly like the room_hint-driven
    version of this scenario already does."""
    project_id = "proj-exec-connection-delete-edit-cluster"
    instance = await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.Furniture.sofa", node_type="Furniture", project_id=project_id, room_id="r1")
    )
    await graph_store.insert_node(
        KnowledgeNode(
            canonical_path="Project.Rooms.r1.Furniture.sofa.Label", node_type="Label", parent_id=instance.node_id,
            value="sofa", project_id=project_id, room_id="r1",
        )
    )
    await graph_store.insert_node(
        KnowledgeNode(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", project_id=project_id, room_id="r1")
    )

    order: list[str] = []

    import app.context_builder as context_builder_module
    import app.versioning as versioning_module

    _real_retract_node = versioning_module.retract_node
    _real_apply_to_graph = context_builder_module.apply_to_graph

    async def tracking_retract_node(node, *args, **kwargs):
        order.append("delete")
        return await _real_retract_node(node, *args, **kwargs)

    async def tracking_apply_to_graph(project_id_, writes):
        if writes:
            order.append("edit")
        return await _real_apply_to_graph(project_id_, writes)

    async def fake_extract_style_only(message, known, **kwargs):
        return ExtractedFields(style="modern")

    async def fake_embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    with (
        patch("app.versioning.retract_node", side_effect=tracking_retract_node),
        patch("app.context_builder.apply_to_graph", side_effect=tracking_apply_to_graph),
        patch("app.llm.extract_fields", side_effect=fake_extract_style_only),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed),
    ):
        result = await execute(
            [
                TaskSpec(type=TaskType.DELETE_CONTEXT, target="remove the sofa", connection="Rooms.r1.Furniture.sofa"),
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="style is modern", connection="Rooms.r1"),
            ],
            base_state("remove the sofa, style is modern", project_id),
        )

    assert order == ["delete", "edit"], f"delete must commit before edit in a connected cluster, got {order}"
    refreshed = await graph_store.find_by_node_id(instance.node_id)
    assert refreshed.lifecycle == "retracted"
    style_node = await graph_store.find_one(project_id, "Project.Rooms.r1.Style")
    assert style_node.value == "modern"
    assert result["needs_answer"] is True


# ---------------------------------------------------------------------------
# Phase 11 hardening — partial-failure isolation, response merging
# ---------------------------------------------------------------------------


async def test_execute_one_failing_task_does_not_lose_the_others_results():
    project_id = "proj-exec-partial-fail"
    with (
        patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("budget is $20k", project_id))

    async def failing_query_catalog(query, limit=5, style_tags=None):
        raise RuntimeError("catalog boom")

    with patch("app.rag.query_catalog", side_effect=failing_query_catalog):
        result = await execute(
            [TaskSpec(type=TaskType.DATABASE_QUERY), TaskSpec(type=TaskType.RETRIEVE_CONTEXT)], base_state("remind me and find a sofa", project_id)
        )

    assert "overallBudget=$20k" in result["retrieved"], "the OTHER task's result must survive one task's failure"
    failed_entries = [t for t in result["trace"] if "failed" in t.output_summary]
    assert len(failed_entries) == 1
    assert "catalog boom" in failed_entries[0].output_summary


async def test_execute_edit_context_failure_does_not_lose_a_concurrent_answer_signal():
    """build_context already has its own internal resilience (extract_fields/
    extract_graph_links failures are absorbed — see app/context_builder.py,
    covered by tests/test_context_builder.py). To exercise execute()'s OWN
    isolation (a dispatched branch failing for some other, unexpected
    reason), patch build_context_node itself rather than the LLM calls
    underneath it."""

    async def failing_build_context_node(state, task=None, resolved=None):
        raise RuntimeError("unexpected boom")

    with (
        patch("app.graph.build_context_node", side_effect=failing_build_context_node),
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT), TaskSpec(type=TaskType.ANSWER)], base_state("modern kitchen, and what would this cost")
        )

    assert "update_summary" not in result, "the failed EDIT_CONTEXT branch must not merge a partial/broken result"
    assert result["needs_answer"] is True, "the batch-level EDIT_CONTEXT flag is set from the task list, not the outcome"
    assert any("unexpected boom" in t.output_summary for t in result["trace"])


# ---------------------------------------------------------------------------
# operation_progress stream events — one per task, as it finishes (see
# execution._progress_event / _stream_writer). Every test above already
# proves execute() works fine with NO writer available (get_stream_writer()
# raises outside a real graph run, exactly the situation every direct
# execute() call in this file is in) — these tests patch get_stream_writer
# itself to prove the events are actually emitted, with the right shape,
# when a writer IS available.
# ---------------------------------------------------------------------------


async def test_execute_emits_one_operation_progress_event_per_task():
    project_id = "proj-exec-progress"
    captured: list[dict] = []

    with (
        patch("app.execution.get_stream_writer", return_value=captured.append),
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
    ):
        await execute(
            [
                TaskSpec(type=TaskType.EDIT_CONTEXT, target="kitchen is modern"),
                TaskSpec(type=TaskType.DATABASE_QUERY, target="find a sofa"),
            ],
            base_state("kitchen is modern, find a sofa", project_id),
        )

    assert len(captured) == 2
    by_type = {e["task_type"]: e for e in captured}
    assert by_type["EDIT_CONTEXT"] == {
        "type": "operation_progress", "task_type": "EDIT_CONTEXT", "target": "kitchen is modern", "ok": True,
    }
    assert by_type["DATABASE_QUERY"] == {
        "type": "operation_progress", "task_type": "DATABASE_QUERY", "target": "find a sofa", "ok": True,
    }


async def test_execute_emits_a_failed_operation_progress_event_for_a_failing_task():
    async def failing_query_catalog(query, limit=5, style_tags=None):
        raise RuntimeError("catalog boom")

    captured: list[dict] = []
    with (
        patch("app.execution.get_stream_writer", return_value=captured.append),
        patch("app.rag.query_catalog", side_effect=failing_query_catalog),
    ):
        await execute([TaskSpec(type=TaskType.DATABASE_QUERY, target="find a sofa")], base_state("find a sofa"))

    assert len(captured) == 1
    assert captured[0]["ok"] is False
    assert captured[0]["task_type"] == "DATABASE_QUERY"


async def test_execute_without_a_graph_run_context_does_not_raise():
    """get_stream_writer() raises RuntimeError outside a real LangGraph run
    (confirmed directly against the real langgraph package, not mocked here)
    — execute() must still complete normally rather than propagating that,
    since every test in this file calls it exactly this way."""
    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("modern kitchen, no graph context"))

    assert result["needs_answer"] is True


async def test_merge_task_results_returns_single_part_unchanged_without_a_model_call():
    with patch("app.llm.merge_response") as mock_merge:
        result = await merge_task_results(["Got it — noted style: modern."])
    assert result == "Got it — noted style: modern."
    mock_merge.assert_not_called()


async def test_merge_task_results_drops_empty_parts_and_returns_empty_string_when_nothing_remains():
    with patch("app.llm.merge_response") as mock_merge:
        result = await merge_task_results(["", ""])
    assert result == ""
    mock_merge.assert_not_called()


async def test_merge_task_results_calls_the_model_for_multiple_real_parts():
    async def fake_merge(parts, **kwargs):
        return "Noted the style, and here's your answer."

    with patch("app.llm.merge_response", side_effect=fake_merge) as mock_merge:
        result = await merge_task_results(["Got it — noted style: modern.", "Oak flooring runs about $8-12/sqft."])

    assert result == "Noted the style, and here's your answer."
    mock_merge.assert_awaited_once_with(["Got it — noted style: modern.", "Oak flooring runs about $8-12/sqft."])
