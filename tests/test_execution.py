from unittest.mock import patch

from app.deepinfra import ExtractedFields, GraphExtraction
from app.execution import execute, merge_task_results
from app.models import KnowledgeNode
from app.tasks import TaskSpec, TaskType


def base_state(message: str, project_id: str = "proj-exec-1") -> dict:
    return {
        "session_id": "test-session",
        "project_id": project_id,
        "message": message,
        "history": "",
        "active_room_id": None,
        "field_attempts": {},
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
        patch("app.deepinfra.extract_fields", side_effect=fake_extract),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("modern kitchen", project_id))

    assert result["needs_answer"] is True
    assert "style: modern" in result["update_summary"]
    style_node = await KnowledgeNode.find_one(
        KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == f"Project.Rooms.{result['active_room_id']}.Style"
    )
    assert style_node.value == "modern"


async def test_execute_single_delete_context_task_retracts_via_delete_context_node():
    project_id = "proj-exec-delete"
    # delete_context_node's matching only searches the five freeform types
    # (see app.canonical_mapper._FREEFORM_NODE_TYPES), not structured slot
    # leaves — a Furniture instance + Label, matching what
    # canonical_mapper._create_instance actually produces.
    instance = KnowledgeNode(canonical_path="Project.Rooms.r1.Furniture.sofa", node_type="Furniture", project_id=project_id, room_id="r1")
    await instance.insert()
    label = KnowledgeNode(
        canonical_path="Project.Rooms.r1.Furniture.sofa.Label", node_type="Label", parent_id=instance.node_id,
        value="sofa", project_id=project_id, room_id="r1",
    )
    await label.insert()

    async def fake_embed(texts):
        # "remove the sofa" (the task's own target text) and "sofa" (the
        # existing Label's value) both need a real, matching non-zero
        # vector — a zero vector's cosine similarity is defined as 0.0 by
        # app.canonical_mapper._cosine_similarity's own guard, which would
        # read as "no match" here rather than a confident one.
        return [[1.0, 0.0, 0.0] for _ in texts]

    with patch("app.canonical_mapper.deepinfra.embed", side_effect=fake_embed):
        result = await execute(
            [TaskSpec(type=TaskType.DELETE_CONTEXT, target="remove the sofa", room_hint="r1")],
            base_state("remove the sofa", project_id),
        )

    assert result["needs_answer"] is True
    assert "sofa" in result["update_summary"]
    refreshed = await KnowledgeNode.find_one(KnowledgeNode.node_id == instance.node_id)
    assert refreshed.lifecycle == "retracted"


async def test_execute_single_database_query_task_returns_retrieved():
    with patch("app.rag.query_catalog", side_effect=fake_query_catalog):
        result = await execute([TaskSpec(type=TaskType.DATABASE_QUERY)], base_state("find a sofa"))

    assert result["retrieved"] == "no matches"
    assert "update_summary" not in result


async def test_execute_retrieve_context_task_summarizes_known_fields():
    project_id = "proj-exec-retrieve"
    with (
        patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
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
        patch("app.deepinfra.extract_fields", side_effect=fake_extract),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        result = await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT), TaskSpec(type=TaskType.ANSWER)],
            base_state("modern kitchen, and what would oak flooring cost", project_id),
        )

    assert result["needs_answer"] is True
    style_node = await KnowledgeNode.find_one(
        KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == f"Project.Rooms.{result['active_room_id']}.Style"
    )
    assert style_node.value == "modern"


async def test_execute_database_query_plus_retrieve_context_merges_both_retrievals():
    project_id = "proj-exec-dq-rc"
    with (
        patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
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
        patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$15k")),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        await execute([TaskSpec(type=TaskType.EDIT_CONTEXT)], base_state("budget is $15k", project_id))

    async def fake_confirmation(field_label, old_value, new_value, **kwargs):
        return f"You said {old_value} before — did you mean to change it to {new_value}?"

    with (
        patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.deepinfra.generate_conflict_confirmation", side_effect=fake_confirmation),
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
# Phase 11 hardening — partial-failure isolation, response merging
# ---------------------------------------------------------------------------


async def test_execute_one_failing_task_does_not_lose_the_others_results():
    project_id = "proj-exec-partial-fail"
    with (
        patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
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

    async def failing_build_context_node(state, task=None):
        raise RuntimeError("unexpected boom")

    with patch("app.graph.build_context_node", side_effect=failing_build_context_node):
        result = await execute(
            [TaskSpec(type=TaskType.EDIT_CONTEXT), TaskSpec(type=TaskType.ANSWER)], base_state("modern kitchen, and what would this cost")
        )

    assert "update_summary" not in result, "the failed EDIT_CONTEXT branch must not merge a partial/broken result"
    assert result["needs_answer"] is True, "the batch-level EDIT_CONTEXT flag is set from the task list, not the outcome"
    assert any("failed" in t.output_summary for t in result["trace"])


async def test_merge_task_results_returns_single_part_unchanged_without_a_model_call():
    with patch("app.deepinfra.merge_response") as mock_merge:
        result = await merge_task_results(["Got it — noted style: modern."])
    assert result == "Got it — noted style: modern."
    mock_merge.assert_not_called()


async def test_merge_task_results_drops_empty_parts_and_returns_empty_string_when_nothing_remains():
    with patch("app.deepinfra.merge_response") as mock_merge:
        result = await merge_task_results(["", ""])
    assert result == ""
    mock_merge.assert_not_called()


async def test_merge_task_results_calls_the_model_for_multiple_real_parts():
    async def fake_merge(parts, **kwargs):
        return "Noted the style, and here's your answer."

    with patch("app.deepinfra.merge_response", side_effect=fake_merge) as mock_merge:
        result = await merge_task_results(["Got it — noted style: modern.", "Oak flooring runs about $8-12/sqft."])

    assert result == "Noted the style, and here's your answer."
    mock_merge.assert_awaited_once_with(["Got it — noted style: modern.", "Oak flooring runs about $8-12/sqft."])
