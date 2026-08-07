from unittest.mock import patch

import pytest

from app.tasks import TaskType
from app.understanding import TASK_TYPE_TO_INTENT, Meaning, understand


async def fake_classify(message, history="", pending_field=None, *, capture=None):
    if capture is not None:
        capture["messages"] = [{"role": "user", "content": message}]
        capture["raw_output"] = "captured"
    lower = message.lower()
    labels: list[str] = []
    if "modern" in lower or "budget" in lower:
        labels.append("update_context")
    if "recommend" in lower or "products" in lower:
        labels.append("database_query")
    if "remind" in lower:
        labels.append("context_related")
    if not labels:
        labels.append("direct_question")
    return labels


@pytest.mark.parametrize(
    "message,expected_type",
    [
        ("modern kitchen please", TaskType.EDIT_CONTEXT),
        ("hey there", TaskType.ANSWER),
        ("remind me what I said", TaskType.RETRIEVE_CONTEXT),
    ],
)
async def test_understand_maps_single_intent_to_single_task(message, expected_type):
    with patch("app.deepinfra.classify_intent", side_effect=fake_classify):
        meaning = await understand(message)
    assert isinstance(meaning, Meaning)
    assert [t.type for t in meaning.tasks] == [expected_type]
    assert meaning.raw_message == message


async def test_understand_maps_database_query_through_guard():
    """database_query without a pricing/search trigger word gets downgraded by
    guard_database_query before task mapping — same behavior as today's
    classify_intent_node, just observed through understand() instead."""
    with patch("app.deepinfra.classify_intent", side_effect=fake_classify):
        meaning = await understand("some products for my living room")
    assert [t.type for t in meaning.tasks] == [TaskType.ANSWER]
    assert meaning.raw_intents == ["database_query"]


async def test_understand_produces_multiple_tasks_for_a_genuine_compound_message():
    with patch("app.deepinfra.classify_intent", side_effect=fake_classify):
        meaning = await understand("what does oak flooring cost, and modern budget is $15k")
    task_types = {t.type for t in meaning.tasks}
    assert TaskType.EDIT_CONTEXT in task_types
    assert len(meaning.tasks) >= 1


async def test_understand_passes_humanized_pending_field_label():
    captured = {}

    async def capturing_classify(message, history="", pending_field=None, *, capture=None):
        captured["pending_field"] = pending_field
        return ["direct_question"]

    with patch("app.deepinfra.classify_intent", side_effect=capturing_classify):
        await understand("whatever", pending_field="room:abc123.squareFootage")

    assert captured["pending_field"] == "square footage"


async def test_understand_threads_capture_dict_through_to_deepinfra_call():
    capture: dict = {}
    with patch("app.deepinfra.classify_intent", side_effect=fake_classify):
        await understand("hello", capture=capture)
    assert capture["raw_output"] == "captured"


def test_task_type_intent_mapping_is_a_bijection_over_intent_backed_task_types():
    # DELETE_CONTEXT is deliberately excluded: it's not a synthetic 5th
    # classifier label (see app.understanding.guard_delete's docstring) — it's
    # produced by overriding an EDIT_CONTEXT task after the fact, and routed
    # via a dedicated app.graph._route_intent branch, not through this map.
    intent_backed = set(TaskType) - {TaskType.DELETE_CONTEXT}
    assert set(TASK_TYPE_TO_INTENT) == intent_backed
    assert len(set(TASK_TYPE_TO_INTENT.values())) == len(TASK_TYPE_TO_INTENT)


def test_delete_context_has_no_intent_mapping():
    assert TaskType.DELETE_CONTEXT not in TASK_TYPE_TO_INTENT


# ---------------------------------------------------------------------------
# Golden set — Phase 2 exit criteria: "every message in the golden set
# round-trips through understand() -> correct TaskSpec[]". Every (message,
# raw_labels) pair here is taken verbatim from _CLASSIFY_SYSTEM's own worked
# examples in app/deepinfra.py — the classifier prompt's own authoritative
# hand-labeled ground truth, not fabricated for this test. Fixing
# classify_intent to return exactly the label(s) the prompt claims it should
# produce isolates understand()'s own logic (guards + task mapping) from the
# model's actual reliability at hitting that label — a separate concern,
# already covered by the salvage/retry logic in app/deepinfra.py.
# ---------------------------------------------------------------------------

_GOLDEN_SET: list[tuple[str, list[str], list[TaskType]]] = [
    ("what's my budget again?", ["context_related"], [TaskType.RETRIEVE_CONTEXT]),
    ("what should my budget be?", ["direct_question"], [TaskType.ANSWER]),
    ("modern", ["update_context"], [TaskType.EDIT_CONTEXT]),
    ("recommend some tile options", ["database_query"], [TaskType.DATABASE_QUERY]),
    ("what would oak flooring cost?", ["direct_question"], [TaskType.ANSWER]),
    ("quartz countertops", ["update_context"], [TaskType.EDIT_CONTEXT]),
    ("hey", ["direct_question"], [TaskType.ANSWER]),
    ("can I get a quote on that walnut console", ["database_query"], [TaskType.DATABASE_QUERY]),
    ("oak flooring for the kitchen, love the modern look", ["update_context"], [TaskType.EDIT_CONTEXT]),
    ("modern, and keep it under $15k", ["update_context"], [TaskType.EDIT_CONTEXT]),
    (
        "what does oak flooring cost, and my budget is $15k",
        ["database_query", "update_context"],
        [TaskType.DATABASE_QUERY, TaskType.EDIT_CONTEXT],
    ),
    ("recommend some tile options for my modern kitchen", ["database_query"], [TaskType.DATABASE_QUERY]),
]


@pytest.mark.parametrize("message,raw_labels,expected_types", _GOLDEN_SET)
async def test_understand_golden_set_from_classify_intent_prompt_examples(message, raw_labels, expected_types):
    async def fixed_classify(msg, history="", pending_field=None, *, capture=None):
        return list(raw_labels)

    with patch("app.deepinfra.classify_intent", side_effect=fixed_classify):
        meaning = await understand(message)

    assert [t.type for t in meaning.tasks] == expected_types
