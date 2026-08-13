from unittest.mock import patch

import pytest

from app.llm import Operation
from app.tasks import TaskType
from app.understanding import TASK_TYPE_TO_INTENT, Meaning, understand


async def fake_classify_operations(message, history="", pending_field=None, *, tree_text=None, capture=None):
    if capture is not None:
        capture["messages"] = [{"role": "user", "content": message}]
        capture["raw_output"] = "captured"
    lower = message.lower()
    operations: list[Operation] = []
    if "modern" in lower or "budget" in lower:
        operations.append(Operation(text=message, intent="CONTEXT_UPDATE"))
    if "recommend" in lower or "products" in lower:
        operations.append(Operation(text=message, intent="DATABASE_RETRIEVAL"))
    if "remind" in lower:
        operations.append(Operation(text=message, intent="CONTEXT_RETRIEVAL"))
    if "remove" in lower or "delete" in lower:
        operations.append(Operation(text=message, intent="CONTEXT_DELETE"))
    if not operations:
        operations.append(Operation(text=message, intent="DIRECT_ANSWER"))
    return operations


@pytest.mark.parametrize(
    "message,expected_type",
    [
        ("modern kitchen please", TaskType.EDIT_CONTEXT),
        ("hey there", TaskType.ANSWER),
        ("remind me what I said", TaskType.RETRIEVE_CONTEXT),
        ("remove the ceiling fan", TaskType.DELETE_CONTEXT),
        ("recommend some tiles", TaskType.DATABASE_QUERY),
    ],
)
async def test_understand_maps_single_intent_to_single_task(message, expected_type):
    with patch("app.llm.classify_operations", side_effect=fake_classify_operations):
        meaning = await understand(message)
    assert isinstance(meaning, Meaning)
    assert [t.type for t in meaning.tasks] == [expected_type]
    assert meaning.raw_message == message


async def test_understand_produces_one_task_per_operation_for_a_compound_message():
    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(text="what does oak flooring cost", intent="DATABASE_RETRIEVAL"),
            Operation(text="the budget is $15k", intent="CONTEXT_UPDATE"),
        ]

    with patch("app.llm.classify_operations", side_effect=fixed_classify):
        meaning = await understand("what does oak flooring cost, and my budget is $15k")

    assert [t.type for t in meaning.tasks] == [TaskType.DATABASE_QUERY, TaskType.EDIT_CONTEXT]
    assert [t.target for t in meaning.tasks] == ["what does oak flooring cost", "the budget is $15k"]


async def test_understand_no_longer_applies_any_guard_downgrade():
    """A database-flavored operation with no pricing/search trigger word used
    to get downgraded to direct_question by guard_database_query — that guard
    is no longer part of understand()'s path (classify_operations decides the
    intent directly, per its own RULE 10)."""
    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [Operation(text=message, intent="DATABASE_RETRIEVAL")]

    with patch("app.llm.classify_operations", side_effect=fixed_classify):
        meaning = await understand("some products for my living room")

    assert [t.type for t in meaning.tasks] == [TaskType.DATABASE_QUERY]
    assert meaning.raw_intents == ["DATABASE_RETRIEVAL"]


async def test_understand_passes_humanized_pending_field_label():
    captured = {}

    async def capturing_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        captured["pending_field"] = pending_field
        return [Operation(text=message, intent="DIRECT_ANSWER")]

    with patch("app.llm.classify_operations", side_effect=capturing_classify):
        await understand("whatever", pending_field=["room:abc123.squareFootage"])

    assert captured["pending_field"] == "square footage"


async def test_understand_joins_multiple_pending_field_labels_from_a_batched_question():
    captured = {}

    async def capturing_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        captured["pending_field"] = pending_field
        return [Operation(text=message, intent="DIRECT_ANSWER")]

    with patch("app.llm.classify_operations", side_effect=capturing_classify):
        await understand(
            "whatever",
            pending_field=["Project.Rooms.abc123.Style", "Project.Rooms.abc123.SquareFootage"],
        )

    assert captured["pending_field"] == "style, square footage"


async def test_understand_threads_capture_dict_through_to_llm_call():
    capture: dict = {}
    with patch("app.llm.classify_operations", side_effect=fake_classify_operations):
        await understand("hello", capture=capture)
    assert capture["raw_output"] == "captured"


async def test_understand_falls_back_to_direct_answer_on_empty_operations():
    """RULE 15 of classify_operations' own prompt: unclear/no-op input returns
    an empty operations list. understand() must still produce a reply-able
    turn rather than an empty task list."""
    async def empty_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return []

    with patch("app.llm.classify_operations", side_effect=empty_classify):
        meaning = await understand("ok thanks")

    assert [t.type for t in meaning.tasks] == [TaskType.ANSWER]
    assert meaning.tasks[0].target == "ok thanks"
    assert meaning.raw_intents == ["DIRECT_ANSWER"]


async def test_understand_never_auto_derives_room_hint_anymore():
    """The old regex-based room_hint auto-detector (static vocabulary
    matching against an operation's own text) is removed as of the
    classifier-connection plan's Phase 5 cleanup — op.connection (sourced
    from classify_operations' own project-grounded CURRENT DATA TREE
    reasoning) strictly subsumes what it used to offer. room_hint is now
    always None straight out of understand(), for both multi-op and
    single-op turns alike; app.execution/app.graph derive it locally, per
    task, from connection instead (see canonical_mapper.split_connection)."""
    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(text="the living room needs a walnut TV unit", intent="CONTEXT_UPDATE"),
            Operation(text="the kitchen should have white acrylic cabinets", intent="CONTEXT_UPDATE"),
        ]

    with patch("app.llm.classify_operations", side_effect=fixed_classify):
        meaning = await understand(
            "the living room needs a walnut TV unit and the kitchen should have white acrylic cabinets"
        )

    assert [t.room_hint for t in meaning.tasks] == [None, None]


async def test_understand_carries_op_id_and_connection_onto_task_spec():
    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(id="op_1", text="change the countertop to granite", intent="CONTEXT_UPDATE", connection="Rooms.a1b2c3d4.Materials.countertop"),
            Operation(id="op_2", text="change the cabinet to walnut", intent="CONTEXT_UPDATE", connection=None),
        ]

    with patch("app.llm.classify_operations", side_effect=fixed_classify):
        meaning = await understand("change the countertop to granite and the cabinet to walnut")

    assert [t.op_id for t in meaning.tasks] == ["op_1", "op_2"]
    assert meaning.tasks[0].connection == "Rooms.a1b2c3d4.Materials.countertop"
    assert meaning.tasks[1].connection is None


def test_task_type_intent_mapping_is_a_bijection_over_every_task_type():
    # Unlike the old classify_intent taxonomy, CONTEXT_DELETE is a real
    # classifier label now (not a regex override applied after task
    # mapping — see app.understanding.guard_delete, kept only for its own
    # direct tests) — so every TaskType has an intent now.
    assert set(TASK_TYPE_TO_INTENT) == set(TaskType)
    assert len(set(TASK_TYPE_TO_INTENT.values())) == len(TASK_TYPE_TO_INTENT)


def test_delete_context_maps_to_context_delete_intent():
    assert TASK_TYPE_TO_INTENT[TaskType.DELETE_CONTEXT] == "CONTEXT_DELETE"


# ---------------------------------------------------------------------------
# Golden set — every (message, operations) pair here is taken verbatim from
# CLASSIFY_OPERATIONS_SYSTEM's own worked examples in app/prompts.py.
# Fixing classify_operations to return exactly what the prompt claims it
# should isolates understand()'s own logic (intent-to-TaskType mapping, room
# hint detection) from the model's actual reliability at hitting that split —
# a separate concern already covered by the salvage/retry tests in
# tests/test_llm.py.
# ---------------------------------------------------------------------------

_GOLDEN_SET: list[tuple[str, list[tuple[str, str]], list[TaskType]]] = [
    ("The project is a 3BHK apartment.", [("The project is a 3BHK apartment.", "CONTEXT_UPDATE")], [TaskType.EDIT_CONTEXT]),
    ("Remove the TV unit.", [("Remove the TV unit.", "CONTEXT_DELETE")], [TaskType.DELETE_CONTEXT]),
    (
        "What is the project budget?",
        [("What is the project budget?", "CONTEXT_RETRIEVAL")],
        [TaskType.RETRIEVE_CONTEXT],
    ),
    ("Show me walnut finishes.", [("Show me walnut finishes.", "DATABASE_RETRIEVAL")], [TaskType.DATABASE_QUERY]),
    ("What is MDF?", [("What is MDF?", "DIRECT_ANSWER")], [TaskType.ANSWER]),
    (
        "The living room needs a walnut TV unit and the kitchen should have white acrylic cabinets.",
        [
            ("The living room needs a walnut TV unit", "CONTEXT_UPDATE"),
            ("The kitchen should have white acrylic cabinets", "CONTEXT_UPDATE"),
        ],
        [TaskType.EDIT_CONTEXT, TaskType.EDIT_CONTEXT],
    ),
    (
        "I want a modern living room, and what budget did we decide?",
        [
            ("I want a modern living room", "CONTEXT_UPDATE"),
            ("what budget did we decide?", "CONTEXT_RETRIEVAL"),
        ],
        [TaskType.EDIT_CONTEXT, TaskType.RETRIEVE_CONTEXT],
    ),
    (
        "What material did we choose for the kitchen, and show me similar materials?",
        [
            ("What material did we choose for the kitchen", "CONTEXT_RETRIEVAL"),
            ("show me similar materials", "DATABASE_RETRIEVAL"),
        ],
        [TaskType.RETRIEVE_CONTEXT, TaskType.DATABASE_QUERY],
    ),
]


@pytest.mark.parametrize("message,operations,expected_types", _GOLDEN_SET)
async def test_understand_golden_set_from_classify_operations_prompt_examples(message, operations, expected_types):
    async def fixed_classify(msg, history="", pending_field=None, *, tree_text=None, capture=None):
        return [Operation(text=t, intent=i) for t, i in operations]

    with patch("app.llm.classify_operations", side_effect=fixed_classify):
        meaning = await understand(message)

    assert [t.type for t in meaning.tasks] == expected_types
    assert [t.target for t in meaning.tasks] == [t for t, _ in operations]
