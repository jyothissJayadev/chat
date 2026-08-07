"""Covers extract_fields' salvage path (app/deepinfra.py): this small Turbo
model's first tool-call attempt is frequently correct but arrives as its own
native '<function=...>' text syntax instead of a real tool call, which
instructor's strict TOOLS mode rejects — and retries after that degrade to
empty completions rather than recovering (live-observed, see graph.py issue
that motivated this). Without a fallback, that turn's real answer was
silently discarded."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from instructor.core.exceptions import InstructorRetryException
from instructor.v2.core.errors import FailedAttempt

from app.deepinfra import (
    ExtractedFields,
    GraphExtraction,
    GraphNodeRevise,
    classify_intent,
    extract_fields,
    extract_graph_links,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _retry_exc(contents: list[str]) -> InstructorRetryException:
    attempts = [
        FailedAttempt(attempt_number=i + 1, exception=ValueError("boom"), completion=_FakeCompletion(c))
        for i, c in enumerate(contents)
    ]
    return InstructorRetryException(
        last_completion=attempts[-1].completion if attempts else None,
        messages=[],
        n_attempts=len(contents),
        total_usage=0,
        failed_attempts=attempts,
    )


class _FakeFunctionCall:
    def __init__(self, arguments):
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, arguments):
        self.function = _FakeFunctionCall(arguments)


class _FakeToolCallMessage:
    """A real (non-text-syntax) tool call, as instructor's TOOLS mode stores
    it — .content is empty, the payload lives in .tool_calls, same shape as
    the production trace that motivated the type-coercion salvage path."""

    def __init__(self, arguments):
        self.content = None
        self.tool_calls = [_FakeToolCall(arguments)]


class _FakeToolCallChoice:
    def __init__(self, arguments):
        self.message = _FakeToolCallMessage(arguments)


class _FakeToolCallCompletion:
    def __init__(self, arguments):
        self.choices = [_FakeToolCallChoice(arguments)]


def _retry_exc_with_tool_calls(arguments_list: list[str]) -> InstructorRetryException:
    attempts = [
        FailedAttempt(attempt_number=i + 1, exception=ValueError("boom"), completion=_FakeToolCallCompletion(a))
        for i, a in enumerate(arguments_list)
    ]
    return InstructorRetryException(
        last_completion=attempts[-1].completion if attempts else None,
        messages=[],
        n_attempts=len(arguments_list),
        total_usage=0,
        failed_attempts=attempts,
    )


async def test_extract_fields_salvages_text_mode_function_call():
    exc = _retry_exc(
        [
            '<function=ExtractedFields>{"projectType": "residential", "roomType": null}</function>',
        ]
    )
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result = await extract_fields("residential", {})

    assert result.projectType == "residential"
    assert result.roomType is None

    # Salvaged straight off the single cheap first attempt — no need to have
    # spent the full retry budget to recover an answer that was already
    # right, just wrapped in the wrong syntax.
    assert mock_create.call_count == 1
    assert mock_create.call_args.kwargs["max_retries"] == 0


async def test_extract_fields_falls_back_to_full_retry_budget_when_first_attempt_unsalvageable():
    exc = _retry_exc(["", "", "not json at all"])
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        with pytest.raises(InstructorRetryException):
            await extract_fields("residential", {})

    # Cheap first attempt, then one fallback call spending the rest of the
    # original retry budget — total ceiling unchanged (1 + 8 = 9 attempts).
    assert mock_create.call_count == 2
    assert [c.kwargs["max_retries"] for c in mock_create.call_args_list] == [0, 7]


async def test_extract_fields_salvage_warns_on_duplicate_key_and_keeps_last_value(caplog):
    """Reproduces a production data-loss bug found via Langfuse: given a
    message stating both a project-wide total ('15 lakh') and a room-specific
    figure ('4 lakh'), the model emitted 'budgetOrRequirement' twice instead
    of routing the total to overallBudget — standard JSON parsing silently
    keeps only the last value, so the 15 lakh total vanished with no error.
    That specific confusion is now addressed at the prompt level (see
    ExtractedFields.overallBudget/budgetOrRequirement's descriptions), but
    duplicate keys generally can still occur — this locks in that the salvage
    path at least surfaces it via a warning instead of staying silent."""
    exc = _retry_exc(
        [
            '<function=ExtractedFields>{"projectType": "residential", '
            '"budgetOrRequirement": "15 lakh", "roomType": "kitchen", '
            '"budgetOrRequirement": "4 lakh"}</function>',
        ]
    )
    mock_create = AsyncMock(side_effect=exc)

    with caplog.at_level("WARNING", logger="app.deepinfra"):
        with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
            result = await extract_fields("15 lakh total, 4 lakh for the kitchen", {})

    assert result.budgetOrRequirement == "4 lakh", "documents current (lossy) behavior — last value wins"
    assert any("duplicate key" in r.message.lower() for r in caplog.records)
    assert any("budgetOrRequirement" in r.message for r in caplog.records)


async def test_classify_intent_populates_capture_with_messages_and_raw_output():
    mock_create = AsyncMock(return_value=_FakeCompletion("update_context"))
    capture: dict = {}
    with patch("app.deepinfra.client.chat.completions.create", mock_create):
        await classify_intent("modern kitchen", capture=capture)

    assert capture["messages"][0]["role"] == "system"
    assert capture["messages"][1]["content"].endswith("modern kitchen")
    assert capture["raw_output"] == "update_context"


async def test_extract_fields_populates_capture_on_clean_success():
    mock_create = AsyncMock(return_value=(ExtractedFields(style="modern"), _FakeCompletion('{"style": "modern"}')))
    capture: dict = {}
    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result = await extract_fields("modern style", {}, capture=capture)

    assert result.style == "modern"
    assert capture["messages"][1]["content"].endswith("modern style")
    assert capture["raw_output"] == '{"style": "modern"}'


async def test_extract_fields_populates_capture_on_salvage():
    exc = _retry_exc(['<function=ExtractedFields>{"projectType": "residential"}</function>'])
    capture: dict = {}
    with patch(
        "app.deepinfra.structured_client.chat.completions.create_with_completion", AsyncMock(side_effect=exc)
    ):
        result = await extract_fields("residential", {}, capture=capture)

    assert result.projectType == "residential"
    assert capture["messages"] is not None
    assert capture["raw_output"] == '{"projectType": "residential"}'


async def test_classify_intent_includes_pending_field_in_prompt():
    mock_create = AsyncMock(return_value=_FakeCompletion("update_context"))
    with patch("app.deepinfra.client.chat.completions.create", mock_create):
        await classify_intent("modern", history="assistant: what style?", pending_field="style")

    user_content = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "Currently pending question (if any): style" in user_content


async def test_classify_intent_renders_missing_pending_field_as_none():
    mock_create = AsyncMock(return_value=_FakeCompletion("direct_question"))
    with patch("app.deepinfra.client.chat.completions.create", mock_create):
        await classify_intent("hello")

    user_content = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "Currently pending question (if any): none" in user_content


async def test_extract_graph_links_includes_anchors_and_recent_window_in_prompt():
    mock_create = AsyncMock(return_value=(GraphExtraction(), _FakeCompletion("{}")))
    anchors = [{"id": "room:8f3a1c2d", "label": "Kitchen", "type": "room"}]
    recent_nodes = [{"id": "kitchen_cabinet", "label": "Kitchen cabinet", "type": "entity"}]
    recent_edges = [{"source": "kitchen_cabinet", "target": "room:8f3a1c2d", "relation": "located_in"}]

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result, status = await extract_graph_links("acrylic finish", anchors, recent_nodes, recent_edges)

    assert status == "clean"
    assert result == GraphExtraction()
    user_content = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "room:8f3a1c2d (room): Kitchen" in user_content
    assert "kitchen_cabinet -[located_in]-> room:8f3a1c2d" in user_content


async def test_extract_graph_links_renders_empty_anchors_and_window_as_placeholders():
    mock_create = AsyncMock(return_value=(GraphExtraction(), _FakeCompletion("{}")))

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        await extract_graph_links("acrylic finish", [], [], [])

    user_content = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "Known anchors" in user_content and "(none yet)" in user_content
    assert "Recently mentioned items:\n(none recent)" in user_content
    assert "Recent relations:\n(none recent)" in user_content


async def test_extract_graph_links_salvages_text_mode_function_call():
    exc = _retry_exc(
        [
            '<function=GraphExtraction>{"new_nodes": [{"id": "style_modern", '
            '"label": "Modern style", "type": "preference"}], "new_edges": []}</function>',
        ]
    )
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result, status = await extract_graph_links("modern", [], [], [])

    assert status == "salvaged"
    assert result.new_nodes[0].id == "style_modern"
    assert mock_create.call_count == 1
    assert mock_create.call_args.kwargs["max_retries"] == 0


async def test_extract_graph_links_falls_back_and_raises_when_unsalvageable():
    exc = _retry_exc(["", "not json at all"])
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        with pytest.raises(InstructorRetryException):
            await extract_graph_links("modern", [], [], [])

    assert mock_create.call_count == 2
    assert [c.kwargs["max_retries"] for c in mock_create.call_args_list] == [0, 4]


async def test_extract_graph_links_coerces_anchor_only_type_on_real_tool_call():
    """Reproduces a production failure caught via Langfuse: the model emits a
    real, well-formed tool call but tags a new node 'budget' — a type it saw
    on the ANCHOR nodes it was shown, but that GraphNodeCreate doesn't offer
    for new nodes. Retrying verbatim doesn't help (the model repeated the
    identical mistake on all 6 attempts in production, burning 81s for a
    result that got thrown away). Salvage must coerce the type and recover on
    the very first failed attempt instead of exhausting the retry budget."""
    bad_arguments = json.dumps(
        {
            "new_nodes": [{"id": "kitchen_budget", "label": "Kitchen budget", "type": "budget"}],
            "new_edges": [{"source": "kitchen_budget", "target": "project", "relation": "budget_for"}],
        }
    )
    exc = _retry_exc_with_tool_calls([bad_arguments])
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result, status = await extract_graph_links("4 lakh for the kitchen", [], [], [])

    assert status == "salvaged"
    assert result.new_nodes[0].id == "kitchen_budget"
    assert result.new_nodes[0].type == "attribute", "anchor-only type must be coerced to the safe default"
    assert result.new_edges[0].relation == "budget_for"
    assert mock_create.call_count == 1
    assert mock_create.call_args.kwargs["max_retries"] == 0


async def test_extract_graph_links_system_prompt_forbids_anchor_only_types():
    mock_create = AsyncMock(return_value=(GraphExtraction(), _FakeCompletion("{}")))

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        await extract_graph_links("acrylic finish", [], [], [])

    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "Never 'project' or 'budget'" in system_content


def test_graph_extraction_revise_and_retract_default_to_empty():
    """A response with no corrections must not force the model to emit empty
    lists explicitly — and old cached/salvaged payloads predating these
    fields must still validate."""
    result = GraphExtraction()
    assert result.revised_nodes == []
    assert result.retracted_node_ids == []


def test_graph_extraction_accepts_revise_and_retract_ops():
    result = GraphExtraction(
        revised_nodes=[GraphNodeRevise(target_node_id="attr_a1b2c3d4", new_label="Matte lacquer finish")],
        retracted_node_ids=["attr_accent_wall"],
    )
    assert result.revised_nodes[0].target_node_id == "attr_a1b2c3d4"
    assert result.retracted_node_ids == ["attr_accent_wall"]


async def test_extract_graph_links_salvages_text_mode_function_call_missing_new_fields():
    """The model's own '<function=...>' salvage path (see
    _salvage_extracted_graph_links) must still work when the raw JSON predates
    revised_nodes/retracted_node_ids — those must default to empty rather
    than failing validation."""
    exc = _retry_exc(
        [
            '<function=GraphExtraction>{"new_nodes": [{"id": "style_modern", '
            '"label": "Modern style", "type": "preference"}], "new_edges": []}</function>',
        ]
    )
    mock_create = AsyncMock(side_effect=exc)

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        result, status = await extract_graph_links("modern", [], [], [])

    assert status == "salvaged"
    assert result.revised_nodes == []
    assert result.retracted_node_ids == []


async def test_extract_graph_links_system_prompt_covers_revise_and_retract():
    mock_create = AsyncMock(return_value=(GraphExtraction(), _FakeCompletion("{}")))

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        await extract_graph_links("acrylic finish", [], [], [])

    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "revise_node" in system_content
    assert "not only explicit" in system_content
    assert "retracted_node_ids" in system_content


async def test_extract_fields_system_prompt_preserves_hedged_values():
    from app.deepinfra import ExtractedFields

    mock_create = AsyncMock(return_value=(ExtractedFields(), _FakeCompletion("{}")))
    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        await extract_fields("maybe around 4 lakh for the kitchen", {})

    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "hedge" in system_content.lower()


async def test_extract_graph_links_system_prompt_pins_budget_edge_direction():
    """Reproduces a production inconsistency: the model produced 'project ->
    project_budget' for one budget node and 'kitchen_budget -> kitchen' for
    another in the SAME response — backwards from each other, because the
    prompt described 'budget_for' without ever showing a worked example of
    which side is the source. Locks in that the direction convention and a
    concrete two-figures-in-one-message example are both present."""
    mock_create = AsyncMock(return_value=(GraphExtraction(), _FakeCompletion("{}")))

    with patch("app.deepinfra.structured_client.chat.completions.create_with_completion", mock_create):
        await extract_graph_links("acrylic finish", [], [], [])

    system_content = mock_create.call_args.kwargs["messages"][0]["content"]
    assert "source is the more specific thing" in system_content
    assert "project_budget_total, target: project, relation: budget_for" in system_content
    assert "kitchen_budget, target: 'room:8f3a1c2d', relation: budget_for" in system_content
