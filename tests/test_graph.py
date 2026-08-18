"""Live-turn coverage for app/graph.py, post-KnowledgeNode-cutover (see
ARCHITECTURE_BASELINE.md and the plan at C:\\Users\\jyoth\\.claude\\plans\\imperative-wibbling-floyd.md).

Anchor-scaffold/connectivity-backstop/candidate-search/revise-retract
behavior from the old ContextGraph era is gone — that machinery doesn't
exist anymore. Equivalent coverage for the pieces that DO still exist
(canonical mapping, alias resolution, dedup) lives in
tests/test_canonical_mapper.py and tests/test_context_builder.py; this file
covers what's specific to the live turn: routing, the question/decline/
confirm flow, and end-to-end completion."""

import re
from difflib import SequenceMatcher
from typing import Optional
from unittest.mock import patch

import pytest

from app import graph_store
from app.config import settings
from app.llm import ExtractedFields, GraphExtraction, Operation, RoomResolutionItem, RoomResolutionOption
from app.graph import app_graph, classify_intent_node, retrieve_context_node
from app.models import KnowledgeNode, ProjectContext
from app.tasks import TaskSpec, TaskType

# app.graph.classify_intent_node now holds the turn for ANY write task whose
# connection is still None after classification — single-op turns included,
# no exceptions (see the "always block on touch" follow-up to the
# classifier-connection plan). The real app.llm.classify_operations grounds
# `connection` from the live project tree plus conversation history (see
# app.prompts.CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE's CONNECTION rules); this
# fake approximates just enough of that grounding — an explicit (typo-
# tolerant) room mention, a project-wide fact with no room mention, or a
# continuation message reusing the project's one existing room — so the
# rest of this suite's tests (which are about extraction/dedup/completion,
# not connection-grounding itself) don't spuriously trip the clarification
# gate. Tests that specifically want a genuinely ungrounded op pass their
# own `fixed_classify` instead (see the clarifying-questions tests below).
_ROOM_PHRASES = ("living room", "dining room", "kitchen", "bedroom", "bathroom", "office")
_PROJECT_LEVEL_KEYWORDS = ("budget", "$", "renovation", "timeline", "square", "bhk", "everything")


def _mentioned_room(lower_message: str) -> Optional[str]:
    for phrase in _ROOM_PHRASES:
        if phrase in lower_message:
            return phrase.title()
    for word in re.split(r"[^a-z]+", lower_message):
        for phrase in _ROOM_PHRASES:
            if " " in phrase or not word:
                continue
            if SequenceMatcher(None, word, phrase).ratio() >= 0.8:
                return phrase.title()
    return None


def _sole_existing_room(tree_text: str) -> Optional[str]:
    room_ids = set(re.findall(r"Rooms\.([\w-]+)$", tree_text, re.MULTILINE))
    return next(iter(room_ids)) if len(room_ids) == 1 else None


def _infer_connection(message: str, tree_text: str) -> Optional[str]:
    lower = message.lower()
    room = _mentioned_room(lower)
    if room:
        return room
    if any(k in lower for k in _PROJECT_LEVEL_KEYWORDS):
        return "Project"
    return _sole_existing_room(tree_text) and f"Rooms.{_sole_existing_room(tree_text)}"


async def fake_classify_operations(message, history="", pending_field=None, *, tree_text="Project", **kwargs):
    """Stands in for app.llm.classify_operations — keyword-based, in
    priority order, producing Operation objects tagged with the 5-intent
    taxonomy. Every operation carries the whole message as its text (never a
    sub-clause); tests that need a genuine multi-clause split provide their
    own fixed fake instead (see tests/test_understanding.py's golden set)."""
    lower = message.lower()
    operations: list[Operation] = []
    if any(k in lower for k in ("budget", "modern", "everything", "cozy", "sofa", "floor", "walnut", "kitchen", "renovation", "$")):
        operations.append(Operation(text=message, intent="CONTEXT_UPDATE", connection=_infer_connection(message, tree_text)))
    if "recommend" in lower or "products" in lower:
        operations.append(Operation(text=message, intent="DATABASE_RETRIEVAL"))
    if "remind" in lower or "again" in lower:
        operations.append(Operation(text=message, intent="CONTEXT_RETRIEVAL"))
    if not operations:
        operations.append(Operation(text=message, intent="DIRECT_ANSWER"))
    return operations


async def fake_extract(message, known, **kwargs):
    return ExtractedFields(style="modern")


async def fake_extract_graph_links(message, anchors, recent_nodes, recent_edges, **kwargs):
    return GraphExtraction(), "clean"


async def fake_gen_question(field_name, context, *, is_retry=False, **kwargs):
    prefix = "retry " if is_retry else ""
    return f"{prefix}What's your {field_name}?"


async def fake_generate_wrapup_message(context, **kwargs):
    return "wrapup: all set"


async def fake_answer(message, context, history="", retrieved="", **kwargs):
    for tok in ["Sure", ", ", "here's ", "an ", "answer."]:
        yield tok


async def fake_query_catalog(query, limit=5, style_tags=None):
    return []


async def fake_infer_missing_field(field_name, context, **kwargs):
    return f"assumed-{field_name}"


async def fake_generate_conflict_confirmation(field_label, old_value, new_value, **kwargs):
    return f"You said {old_value} for {field_label} before — now {new_value}?"


async def fake_resolve_room_connections(tree_text, operations, **kwargs):
    """Stands in for app.llm.resolve_room_connections — the ONE batched call
    covering every unresolved op in a turn (replacing the old per-op
    generate_clarification_question). Mirrors the real Room Resolution
    Agent's contract: one result per operation, in order, always carrying a
    question + at least one option (app.prompts.
    ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE rule 15 — there's no "resolvable
    without asking" verdict in this prompt, unlike the old
    ClarificationQuestion it replaces). Folds each operation's own text into
    the question so tests can assert on it. This IS the autouse default (see
    mock_models below) — since every write task with connection=None always
    holds the turn regardless of the model's answer (no confidence escape
    hatch), there's no separate "confident" variant needed anymore."""
    return [
        RoomResolutionItem(
            text=op["text"],
            intent=op["intent"],
            question=f'Which room does "{op["text"]}" refer to?',
            options=[RoomResolutionOption(id="option_1", label="Something else / a new room")],
        )
        for op in operations
    ]


def base_state(message: str, project_id: str = "proj-graph-test") -> dict:
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


async def run(state):
    tokens = []
    final_state = None
    async for mode, chunk in app_graph.astream(state, stream_mode=["custom", "values"]):
        if mode == "custom":
            # "custom" now also carries execution.execute()'s per-operation
            # operation_progress events (see app/execution.py), not just
            # generate_answer's answer_token chunks — only collect the latter.
            if chunk.get("type") == "answer_token":
                tokens.append(chunk["token"])
        else:
            final_state = chunk
    return final_state, "".join(tokens)


async def _nodes(project_id: str) -> dict[str, KnowledgeNode]:
    return {n.canonical_path: n for n in await graph_store.find_nodes(project_id)}


async def _room_id_by_type(project_id: str, room_type: str) -> str:
    """The graph no longer exposes "the active room" in state (active_room_id
    is gone — removed system-wide) — tests that need to know which room_id a
    turn resolved to look it up off the written RoomType node instead."""
    nodes = await _nodes(project_id)
    return next(n.room_id for n in nodes.values() if n.node_type == "RoomType" and n.value == room_type)


@pytest.fixture(autouse=True)
def mock_models():
    with (
        patch("app.llm.classify_operations", side_effect=fake_classify_operations),
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.generate_question", side_effect=fake_gen_question),
        patch("app.llm.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.llm.generate_answer", side_effect=fake_answer),
        patch("app.llm.infer_missing_field", side_effect=fake_infer_missing_field),
        patch("app.llm.generate_conflict_confirmation", side_effect=fake_generate_conflict_confirmation),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        yield


# ---------------------------------------------------------------------------
# Basic routing / structured writes
# ---------------------------------------------------------------------------


async def test_update_context_routes_and_writes_to_knowledge_node():
    state, _ = await run(base_state("I want a modern living room"))
    assert state["intent"] == ["CONTEXT_UPDATE"]
    nodes = await _nodes(state["project_id"])
    style_nodes = [n for n in nodes.values() if n.node_type == "Style"]
    assert len(style_nodes) == 1
    assert style_nodes[0].value == "modern"
    # projectType is checked before any room field, so that's what gets
    # asked about first even though this turn was about a room.
    assert state["pending_question"] == "What's your project type?"
    assert "style: modern" in state["update_summary"]
    node_names = [t.node_name for t in state["trace"]]
    assert node_names[0] == "classify_intent"
    assert "build_context" in node_names
    assert "complete_project" not in node_names


async def test_classify_intent_node_captures_full_llm_prompt_and_output():
    async def fake_classify_with_capture(message, history="", pending_field=None, *, tree_text=None, capture=None):
        if capture is not None:
            capture["messages"] = [{"role": "system", "content": "sys"}, {"role": "user", "content": message}]
            capture["raw_output"] = '{"operations": [...]}'
        return [Operation(text=message, intent="CONTEXT_UPDATE")]

    with patch("app.llm.classify_operations", side_effect=fake_classify_with_capture):
        result = await classify_intent_node(base_state("modern kitchen"))

    entry = result["trace"][0]
    assert entry.node_name == "classify_intent"
    assert entry.llm_input == [{"role": "system", "content": "sys"}, {"role": "user", "content": "modern kitchen"}]
    assert entry.llm_output == '{"operations": [...]}'


async def test_retrieve_context_node_is_rule_based_with_no_llm_capture():
    result = await retrieve_context_node(base_state("modern kitchen"))
    entry = result["trace"][0]
    assert entry.node_name == "retrieve_context"
    assert entry.llm_input is None
    assert entry.llm_output is None


async def test_extract_fields_failure_does_not_break_turn():
    async def failing_extract(message, known, **kwargs):
        raise Exception("No tool calls or function call found in response (mode: TOOLS)")

    with patch("app.llm.extract_fields", side_effect=failing_extract):
        state, _ = await run(base_state("I want a modern living room"))

    assert state["pending_question"] == "What's your project type?"
    trace_by_node = {t.node_name: t for t in state["trace"]}
    assert "extract_entities failed" in trace_by_node["build_context"].output_summary or "0 node(s)" in trace_by_node["build_context"].output_summary


async def test_direct_question_streams_answer():
    state, tokens = await run(base_state("what colors go with navy blue"))
    assert state["intent"] == ["DIRECT_ANSWER"]
    assert state["answer"] == "Sure, here's an answer."
    assert tokens == state["answer"]
    assert state.get("update_summary") is None


# ---------------------------------------------------------------------------
# Operation clarifying-questions (classifier-connection plan)
# ---------------------------------------------------------------------------


async def test_split_intent_holds_for_clarification_when_a_write_task_has_no_connection():
    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(id="op_1", text="change the countertop to granite", intent="CONTEXT_UPDATE", connection="Rooms.r1.Materials.countertop"),
            Operation(id="op_2", text="change the cabinet to walnut", intent="CONTEXT_UPDATE", connection=None),
        ]

    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
    ):
        state, _ = await run(base_state("change the countertop to granite and the cabinet to walnut"))

    pending = state["pending_operation_questions"]
    assert pending is not None
    assert [q["op_id"] for q in pending["questions"]] == ["op_2"]
    assert "cabinet" in pending["questions"][0]["question"]
    assert pending["questions"][0]["options"] == [{"id": "option_1", "label": "Something else / a new room"}]
    assert pending["questions"][0]["allow_custom"] is True
    assert state["question_generated"] is True
    nodes = await _nodes(state["project_id"])
    assert nodes == {}, "nothing should write while any op in the batch is unresolved"


async def test_single_op_turn_also_holds_for_clarification_when_connection_is_null():
    """A SINGLE unresolved write op must hold the turn too, not just a
    multi-op batch — e.g. a bare "add a laminate" with no room in scope:
    classify_operations can't ground it (connection: null), and the graph
    must wait for the user rather than let build_context guess. This used to
    be gated to multi-op turns only; that gate is gone (see
    classify_intent_node's comment on `unresolved`)."""

    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [Operation(id="op_1", text="add a laminate", intent="CONTEXT_UPDATE", connection=None)]

    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
    ):
        state, _ = await run(base_state("add a laminate"))

    pending = state["pending_operation_questions"]
    assert pending is not None
    assert [q["op_id"] for q in pending["questions"]] == ["op_1"]
    assert state["question_generated"] is True
    nodes = await _nodes(state["project_id"])
    assert nodes == {}, "nothing should write while the single op's connection is unresolved"


async def test_single_op_turn_still_holds_when_the_model_resolves_a_single_option():
    """The flip side of the two tests above: even when the Room Resolution
    Agent confidently resolves an unresolved op to a single option (rule 5
    of app.prompts.ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE — "if the room is
    confidently resolved, return EXACTLY ONE option"), the turn must still
    hold. There is no model-confidence escape hatch: every write task the
    classifier itself left ungrounded (connection=None) is always confirmed
    by the user first, never silently applied just because the model only
    offered one option (unlike the old ClarificationQuestion's
    needs_clarification=False verdict, which this prompt has no equivalent
    of at all — every operation always gets a question)."""

    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [Operation(id="op_1", text="I want a modern living room", intent="CONTEXT_UPDATE", connection=None)]

    async def single_option_resolution(tree_text, operations, **kwargs):
        return [
            RoomResolutionItem(
                text=operations[0]["text"],
                intent=operations[0]["intent"],
                question="Where would you like this?",
                options=[RoomResolutionOption(id="r1", label="Living Room")],
            )
        ]

    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify),
        patch("app.llm.resolve_room_connections", side_effect=single_option_resolution),
    ):
        state, _ = await run(base_state("I want a modern living room"))

    pending = state["pending_operation_questions"]
    assert pending is not None
    assert [q["op_id"] for q in pending["questions"]] == ["op_1"]
    assert pending["questions"][0]["options"] == [{"id": "r1", "label": "Living Room"}]
    assert state["question_generated"] is True
    node_names = [t.node_name for t in state["trace"]]
    assert "resolve_room_connections" in node_names
    assert "build_context" not in node_names
    nodes = await _nodes(state["project_id"])
    assert nodes == {}, "nothing should write while the single op's connection is unresolved, even with one confidently-resolved option"


async def test_generate_operation_questions_captures_full_llm_prompt_and_output_for_the_whole_batch():
    """The viewer's Process Trace panel expects real llm_input/llm_output on
    every LLM-calling node's entries (see classify_intent's own capture
    test above) — resolve_room_connections is called ONCE for the WHOLE
    batch of unresolved ops (unlike the old generate_clarification_question,
    called once per op via asyncio.gather), so there is exactly one trace
    entry covering every unresolved op in the turn, not one entry per op."""

    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(id="op_1", text="change the countertop to granite", intent="CONTEXT_UPDATE", connection="Rooms.r1.Materials.countertop"),
            Operation(id="op_2", text="change the cabinet to walnut", intent="CONTEXT_UPDATE", connection=None),
            Operation(id="op_3", text="add a rug", intent="CONTEXT_UPDATE", connection=None),
        ]

    async def fake_resolution_with_capture(tree_text, operations, *, capture=None):
        if capture is not None:
            capture["messages"] = [
                {"role": "system", "content": f"sys for {len(operations)} op(s)"},
                {"role": "user", "content": "Return the JSON array now."},
            ]
            capture["raw_output"] = '{"resolutions": [...]}'
        return [
            RoomResolutionItem(
                text=op["text"],
                intent=op["intent"],
                question=f"Which room for {op['text']}?",
                options=[RoomResolutionOption(id="option_1", label="Kids Bedroom")],
            )
            for op in operations
        ]

    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolution_with_capture),
    ):
        state, _ = await run(base_state("change the countertop to granite, the cabinet to walnut, and add a rug"))

    clarify_entries = [t for t in state["trace"] if t.node_name == "resolve_room_connections"]
    assert len(clarify_entries) == 1, "one batched call covers every unresolved op in the turn, not one call per op"
    entry = clarify_entries[0]
    assert entry.llm_input == [
        {"role": "system", "content": "sys for 2 op(s)"},
        {"role": "user", "content": "Return the JSON array now."},
    ]
    assert entry.llm_output == '{"resolutions": [...]}'
    assert entry.model_used == settings.model_intent_classifier
    pending = state["pending_operation_questions"]
    assert [q["op_id"] for q in pending["questions"]] == ["op_2", "op_3"]


async def test_split_intent_still_holds_when_model_resolves_a_single_option():
    """Same as test_single_op_turn_still_holds_when_the_model_resolves_a_single_option,
    for a multi-op batch: a single-option resolution for op_2 must still
    hold the WHOLE turn back, op_1's own already-resolved connection
    notwithstanding — no model-confidence escape hatch, single-op or
    batched."""

    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [
            Operation(id="op_1", text="change the countertop to granite", intent="CONTEXT_UPDATE", connection="Rooms.r1.Materials.countertop"),
            Operation(id="op_2", text="change the island to walnut", intent="CONTEXT_UPDATE", connection=None),
        ]

    async def single_option_resolution(tree_text, operations, **kwargs):
        return [
            RoomResolutionItem(
                text=operations[0]["text"],
                intent=operations[0]["intent"],
                question="Where is the island?",
                options=[RoomResolutionOption(id="r1", label="Kitchen")],
            )
        ]

    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify),
        patch("app.llm.resolve_room_connections", side_effect=single_option_resolution),
    ):
        state, _ = await run(base_state("change the countertop to granite and the island to walnut"))

    pending = state["pending_operation_questions"]
    assert pending is not None
    assert [q["op_id"] for q in pending["questions"]] == ["op_2"]
    node_names = [t.node_name for t in state["trace"]]
    assert "resolve_room_connections" in node_names
    assert "handle_split_intents" not in node_names
    nodes = await _nodes(state["project_id"])
    assert nodes == {}, "nothing should write while any op in the batch is unresolved, even with one confidently-resolved option"


async def test_classify_intent_node_resumes_from_pending_operation_questions_without_reclassifying():
    reclassify_called = False

    async def should_not_be_called(*args, **kwargs):
        nonlocal reclassify_called
        reclassify_called = True
        return []

    pending_tasks = [
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_1", target="change the countertop to granite", connection="Rooms.r1.Materials.countertop").model_dump(),
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_2", target="change the cabinet to walnut", connection=None).model_dump(),
    ]
    state = base_state("Kids Bedroom")
    state["pending_operation_questions"] = {
        "tasks": pending_tasks,
        "questions": [{"op_id": "op_2", "text": "change the cabinet to walnut", "question": "Which room?", "options": ["Something else / a new room"]}],
    }
    state["operation_answers"] = {"op_2": "Rooms.r2"}

    with patch("app.llm.classify_operations", side_effect=should_not_be_called):
        result = await classify_intent_node(state)

    assert reclassify_called is False
    assert result["pending_operation_questions"] is None
    assert [t.op_id for t in result["tasks"]] == ["op_1", "op_2"]
    assert [t.connection for t in result["tasks"]] == ["Rooms.r1.Materials.countertop", "Rooms.r2"]


async def test_classify_intent_node_reasks_only_still_unresolved_ops_after_a_partial_answer():
    pending_tasks = [
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_1", target="change the cabinet to walnut", connection=None).model_dump(),
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_2", target="add a rug", connection=None).model_dump(),
    ]
    state = base_state("whatever")
    state["pending_operation_questions"] = {
        "tasks": pending_tasks,
        "questions": [
            {"op_id": "op_1", "text": "change the cabinet to walnut", "question": "Which room?", "options": []},
            {"op_id": "op_2", "text": "add a rug", "question": "Which room?", "options": []},
        ],
    }
    state["operation_answers"] = {"op_1": "Rooms.r1"}

    with (
        patch("app.llm.classify_operations", side_effect=AssertionError),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
    ):
        result = await classify_intent_node(state)

    pending = result["pending_operation_questions"]
    assert pending is not None
    assert [q["op_id"] for q in pending["questions"]] == ["op_2"]
    # op_1's already-resolved connection must survive into the re-stashed
    # task list, not just the still-unresolved op_2's.
    assert {t["op_id"]: t["connection"] for t in pending["tasks"]} == {"op_1": "Rooms.r1", "op_2": None}


async def test_classify_intent_node_fans_a_bundled_multi_room_answer_into_one_task_per_room():
    """A bundled option (app.llm.RoomResolutionOption.room_ids) answers ONE
    op_id but names 2+ existing rooms — the resume branch must turn that one
    pending task into one task per room, each independently grounded, rather
    than setting a single connection (see _apply_operation_answers)."""
    pending_tasks = [
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_1", target="add laminate to the wardrobe", connection=None).model_dump(),
    ]
    state = base_state("whatever")
    state["pending_operation_questions"] = {
        "tasks": pending_tasks,
        "questions": [{"op_id": "op_1", "text": "add laminate to the wardrobe", "question": "Which wardrobe?", "options": []}],
    }
    state["operation_answers"] = {"op_1": "Both Bedroom 1 and Bedroom 2"}
    state["operation_room_selections"] = {"op_1": ["r1", "r2"]}

    with patch("app.llm.classify_operations", side_effect=AssertionError):
        result = await classify_intent_node(state)

    assert result["pending_operation_questions"] is None
    assert [t.op_id for t in result["tasks"]] == ["op_1__r1", "op_1__r2"]
    assert [t.connection for t in result["tasks"]] == ["Rooms.r1", "Rooms.r2"]
    assert all(t.target == "add laminate to the wardrobe" for t in result["tasks"])


async def test_classify_intent_node_leaves_non_bundled_answers_alone_in_the_same_resume():
    """A resume batch can mix an ordinary single-room answer with a bundled
    one — only the op_id present in operation_room_selections fans out; the
    other keeps its normal single-connection behavior."""
    pending_tasks = [
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_1", target="change the countertop to granite", connection=None).model_dump(),
        TaskSpec(type=TaskType.EDIT_CONTEXT, op_id="op_2", target="add laminate to the wardrobe", connection=None).model_dump(),
    ]
    state = base_state("whatever")
    state["pending_operation_questions"] = {
        "tasks": pending_tasks,
        "questions": [
            {"op_id": "op_1", "text": "change the countertop to granite", "question": "Which room?", "options": []},
            {"op_id": "op_2", "text": "add laminate to the wardrobe", "question": "Which wardrobe?", "options": []},
        ],
    }
    state["operation_answers"] = {"op_1": "Rooms.r1.Materials.countertop", "op_2": "Both Bedroom 1 and Bedroom 2"}
    state["operation_room_selections"] = {"op_2": ["r1", "r2"]}

    with patch("app.llm.classify_operations", side_effect=AssertionError):
        result = await classify_intent_node(state)

    assert result["pending_operation_questions"] is None
    op_ids = [t.op_id for t in result["tasks"]]
    assert op_ids == ["op_1", "op_2__r1", "op_2__r2"]
    connections = {t.op_id: t.connection for t in result["tasks"]}
    assert connections["op_1"] == "Rooms.r1.Materials.countertop"
    assert connections["op_2__r1"] == "Rooms.r1"
    assert connections["op_2__r2"] == "Rooms.r2"


# ---------------------------------------------------------------------------
# Split-intent concurrency
# ---------------------------------------------------------------------------


async def test_split_intent_with_update_context_answers_and_writes_together():
    state, _ = await run(base_state("what products would you recommend, and I want a modern living room"))

    assert state["intent"] == ["CONTEXT_UPDATE", "DATABASE_RETRIEVAL"]
    nodes = await _nodes(state["project_id"])
    assert any(n.node_type == "Style" and n.value == "modern" for n in nodes.values())
    assert state["answer"] == "Sure, here's an answer."
    assert state["pending_question"] == "What's your project type?"

    node_names = [t.node_name for t in state["trace"]]
    assert "handle_split_intents" in node_names
    assert "query_catalog" in node_names
    assert "build_context" in node_names
    assert "validate_completeness" in node_names
    # generate_question is the single consolidated question-generation node
    # now (the old separate analyze_context_node is merged into it) — it
    # runs twice here: once for real (validate_completeness's "incomplete"
    # branch, producing the actual question), and once more as a guarded
    # no-op pass-through after generate_answer (question_generated is
    # already True by then, so it just returns a trace entry).
    assert node_names.count("generate_question") == 2
    assert node_names.count("generate_answer") == 1


async def test_handle_split_intents_node_produces_its_own_summary_trace_entry():
    """Unlike every other node, handle_split_intents_node used to delegate to
    execution.execute() with no _node_span/trace entry of its own — the
    sub-task entries (build_context, query_catalog, ...) were the only trace
    of the batch ever having run as one step. It now produces one entry
    summarizing the whole batch, ahead of the sub-task entries, matching
    every other node's pattern."""
    state, _ = await run(base_state("what products would you recommend, and I want a modern living room"))

    split_entries = [t for t in state["trace"] if t.node_name == "handle_split_intents"]
    assert len(split_entries) == 1
    entry = split_entries[0]
    assert "2 operation(s)" in entry.output_summary
    assert "EDIT_CONTEXT" in entry.output_summary
    assert "DATABASE_QUERY" in entry.output_summary

    node_names = [t.node_name for t in state["trace"]]
    # The summary entry comes first — it describes the batch as a whole,
    # ahead of what each task inside it did.
    assert node_names.index("handle_split_intents") < node_names.index("build_context")
    assert node_names.index("handle_split_intents") < node_names.index("query_catalog")


async def test_split_intent_without_update_context_merges_both_retrievals():
    state, _ = await run(base_state("remind me what you'd recommend and how much they cost"))

    assert state["intent"] == ["DATABASE_RETRIEVAL", "CONTEXT_RETRIEVAL"]
    assert state["answer"] == "Sure, here's an answer."

    node_names = [t.node_name for t in state["trace"]]
    assert "query_catalog" in node_names
    assert "retrieve_context" in node_names
    assert "validate_completeness" not in node_names
    assert "complete_project" not in node_names
    assert "build_context" not in node_names
    assert node_names.count("generate_answer") == 1


async def test_leftover_question_survives_unrelated_turn():
    state1, _ = await run(base_state("I want a modern living room"))

    state1["message"] = "can you recommend some products and their price"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False

    state2, _ = await run(state1)
    assert state2["intent"] == ["DATABASE_RETRIEVAL"]
    assert state2["pending_question"] == "What's your project type?"
    node_names = [t.node_name for t in state2["trace"]]
    assert node_names.count("classify_intent") == 2
    assert "query_catalog" in node_names


# ---------------------------------------------------------------------------
# Room resolution / materials merge
# ---------------------------------------------------------------------------


async def test_materials_merge_across_turns_by_item():
    project_id = "proj-materials-merge"

    async def fake_extract_flooring(message, known, **kwargs):
        return ExtractedFields(roomType="living room", materials=[{"item": "flooring", "material": "oak wood"}])

    with patch("app.llm.extract_fields", side_effect=fake_extract_flooring):
        state1, _ = await run(base_state("I want oak flooring in the living room", project_id))

    room_id = await _room_id_by_type(project_id, "living room")
    nodes1 = await _nodes(project_id)
    assert nodes1[f"Project.Rooms.{room_id}.Materials.flooring.Material"].value == "oak wood"

    async def fake_extract_sofa(message, known, **kwargs):
        return ExtractedFields(materials=[{"item": "sofa", "material": "leather"}])

    state1["message"] = "make the sofa leather"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_sofa):
        state2, _ = await run(state1)

    nodes2 = await _nodes(project_id)
    assert nodes2[f"Project.Rooms.{room_id}.Materials.flooring.Material"].value == "oak wood", "earlier item must survive"
    assert nodes2[f"Project.Rooms.{room_id}.Materials.sofa.Material"].value == "leather"

    async def fake_extract_flooring_update(message, known, **kwargs):
        return ExtractedFields(materials=[{"item": "flooring", "material": "walnut wood"}])

    state2["message"] = "actually make the flooring walnut instead"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_flooring_update):
        state3, _ = await run(state2)

    nodes3 = await _nodes(project_id)
    assert nodes3[f"Project.Rooms.{room_id}.Materials.flooring.Material"].value == "walnut wood", "same item must be replaced, not duplicated"
    assert nodes3[f"Project.Rooms.{room_id}.Materials.sofa.Material"].value == "leather"


async def test_new_room_mentioned_mid_conversation_does_not_lose_first_room():
    project_id = "proj-new-room-mid"

    async def fake_extract_kitchen(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen", style="modern")

    with patch("app.llm.extract_fields", side_effect=fake_extract_kitchen):
        state1, _ = await run(base_state("the kitchen should be modern", project_id))
    kitchen_id = await _room_id_by_type(project_id, "kitchen")

    async def fake_extract_bedroom(message, known, **kwargs):
        return ExtractedFields(roomType="bedroom", style="cozy")

    state1["message"] = "the bedroom should be cozy too"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_bedroom):
        state2, _ = await run(state1)
    bedroom_id = await _room_id_by_type(project_id, "bedroom")

    assert bedroom_id != kitchen_id
    nodes = await _nodes(project_id)
    assert nodes[f"Project.Rooms.{kitchen_id}.Style"].value == "modern", "first room's data must survive introducing a second room"
    assert nodes[f"Project.Rooms.{bedroom_id}.Style"].value == "cozy"


async def test_typo_respelling_reuses_existing_room_instead_of_forking_a_duplicate():
    project_id = "proj-typo-room"

    async def fake_extract_typo(message, known, **kwargs):
        return ExtractedFields(roomType="bedrroom", style="modern")

    with patch("app.llm.extract_fields", side_effect=fake_extract_typo):
        state1, _ = await run(base_state("the bedrroom should be modern", project_id))
    room_id = await _room_id_by_type(project_id, "bedrroom")

    async def fake_extract_correct_spelling(message, known, **kwargs):
        return ExtractedFields(roomType="bedroom", existingFurniture="none")

    state1["message"] = "add a wardrobe to the bedroom as part of the renovation"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_correct_spelling):
        state2, _ = await run(state1)

    nodes = await _nodes(project_id)
    assert nodes[f"Project.Rooms.{room_id}.Style"].value == "modern"
    assert nodes[f"Project.Rooms.{room_id}.ExistingFurniture"].value == "none", "the respelling must resolve to the existing room, not fork a second one"
    room_count = len({n.room_id for n in nodes.values() if n.room_id})
    assert room_count == 1, "must not fork a duplicate room"


async def test_additional_room_budget_lands_on_the_correct_room_not_the_primary_one():
    project_id = "proj-extra-room-budget"

    async def fake_extract_multi_budget(message, known, **kwargs):
        return ExtractedFields(
            projectType="residential",
            overallBudget="15 lakh",
            roomType="living",
            additionalRoomBudgets=[{"roomType": "kitchen", "budgetOrRequirement": "4 lakh"}],
        )

    with patch("app.llm.extract_fields", side_effect=fake_extract_multi_budget):
        state, _ = await run(base_state("living and kitchen, total 15 lakh, kitchen budget 4 lakh", project_id))

    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "15 lakh"
    living_id = next(n.room_id for n in nodes.values() if n.node_type == "RoomType" and n.value == "living")
    assert f"Project.Rooms.{living_id}.Budget" not in nodes, "the total must never land on the primary room's own budget field"
    kitchen_budget = next(n for path, n in nodes.items() if n.node_type == "Budget" and n.room_id is not None and n.room_id != living_id)
    assert kitchen_budget.value == "4 lakh"


# ---------------------------------------------------------------------------
# Decline / room-skip (see classify_intent_node's decline-detection —
# decline_field_node and the old rephrase-then-infer mechanic are gone
# entirely; a decline now just deprioritizes the whole room into
# skipped_rooms and lets the turn's normally-classified operation proceed).
# ---------------------------------------------------------------------------


def _style_gap_state(project_id: str, room_id: str = "r1") -> dict:
    """A state where the given room's style is the currently pending
    question (current_field is what classify_intent_node's decline-detection
    actually reads — pending_gap is the cache generate_question_node reads/
    writes, see PENDING_GAP_ANALYSIS.md)."""
    state = base_state("not sure", project_id)
    gap = {
        "canonical_path": f"Project.Rooms.{room_id}.Style",
        "field_label": "style",
        "node_type": "Style",
        "room_id": room_id,
    }
    state["pending_gap"] = gap
    state["current_field"] = {"canonical_path": gap["canonical_path"], "room_id": room_id}
    return state


async def _seed_everything_but_style(project_id: str) -> None:
    from app.context_builder import ProposedWrite, apply_to_graph

    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
        ],
    )


async def _seed_two_rooms_missing_style(project_id: str) -> None:
    from app.context_builder import ProposedWrite, apply_to_graph

    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.RoomType", node_type="RoomType", value="bedroom", room_id="r2", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.Budget", node_type="Budget", value="$5k", room_id="r2", tier="critical"),
        ],
    )


async def test_decline_skips_the_whole_room_and_moves_to_a_different_room():
    project_id = "proj-decline-skip"
    await _seed_two_rooms_missing_style(project_id)
    state = _style_gap_state(project_id, "r1")

    state1, _ = await run(state)

    assert state1["skipped_rooms"] == ["r1"]
    nodes = await _nodes(project_id)
    assert "Project.Rooms.r1.Style" not in nodes, "declining must not fabricate a value"
    # r1 is now skipped, so the next question must move on to r2's style,
    # not re-ask about r1's.
    assert state1["current_field"] == {"canonical_path": "Project.Rooms.r2.Style", "room_id": "r2"}
    assert state1["pending_gap"]["canonical_path"] == "Project.Rooms.r2.Style"
    assert "style" in state1["pending_question"].lower()
    node_names = [t.node_name for t in state1["trace"]]
    assert "decline_field" not in node_names, "decline_field_node no longer exists"


async def test_skipped_room_is_revisited_only_after_every_other_room_is_asked_through():
    project_id = "proj-decline-revisit"
    await _seed_two_rooms_missing_style(project_id)
    state = _style_gap_state(project_id, "r1")

    state1, _ = await run(state)
    assert state1["current_field"]["room_id"] == "r2"

    async def fixed_classify_r2(message, history="", pending_field=None, *, tree_text=None, capture=None):
        return [Operation(id="op_1", text=message, intent="CONTEXT_UPDATE", connection="Rooms.r2")]

    async def fake_extract_style(message, known, **kwargs):
        return ExtractedFields(style="modern")

    state1["message"] = "modern"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify_r2),
        patch("app.llm.extract_fields", side_effect=fake_extract_style),
    ):
        state2, _ = await run(state1)

    # r2's style is filled; r2's next open field (square footage) is asked
    # next — r1 (still skipped) is not revisited yet.
    assert state2["current_field"]["room_id"] == "r2"
    assert "squarefootage" in state2["pending_gap"]["canonical_path"].lower()

    async def fake_extract_sqft(message, known, **kwargs):
        return ExtractedFields(squareFootage=120)

    state2["message"] = "120 sqft"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify_r2),
        patch("app.llm.extract_fields", side_effect=fake_extract_sqft),
    ):
        state3, _ = await run(state2)

    assert "existingfurniture" in state3["pending_gap"]["canonical_path"].lower()

    async def fake_extract_furniture(message, known, **kwargs):
        return ExtractedFields(existingFurniture="none")

    state3["message"] = "none"
    state3["intent"] = []
    state3["pending_question"] = None
    state3["question_generated"] = False
    with (
        patch("app.llm.classify_operations", side_effect=fixed_classify_r2),
        patch("app.llm.extract_fields", side_effect=fake_extract_furniture),
    ):
        state4, _ = await run(state3)

    # r2 is now fully resolved — r1 (skipped) is the only room left with
    # anything open, so it's revisited automatically.
    assert state4["current_field"]["room_id"] == "r1"
    assert state4["pending_gap"]["canonical_path"] == "Project.Rooms.r1.Style"


async def test_declining_a_non_room_scoped_gap_just_reasks_unchanged():
    """ProjectType/Rooms-existence/Budget.Total have no room to skip — a
    decline is simply a no-op and the same question is re-asked, unchanged."""
    project_id = "proj-decline-project-level"
    state = base_state("not sure", project_id)
    gap = {
        "canonical_path": "Project.BasicInformation.ProjectType",
        "field_label": "project type",
        "node_type": "ProjectType",
        "room_id": None,
    }
    state["pending_gap"] = gap
    state["current_field"] = {"canonical_path": gap["canonical_path"], "room_id": None}

    state1, _ = await run(state)

    assert state1["skipped_rooms"] == []
    assert state1["pending_question"] == "What's your project type?"
    assert state1["current_field"] == {"canonical_path": "Project.BasicInformation.ProjectType", "room_id": None}


async def test_unrelated_answer_is_not_treated_as_a_decline():
    project_id = "proj-decline-unrelated"
    await _seed_everything_but_style(project_id)
    state = _style_gap_state(project_id)
    state["message"] = "modern with a lot of natural light"

    with patch("app.llm.extract_fields", side_effect=fake_extract):
        state1, _ = await run(state)

    assert state1["intent"] == ["CONTEXT_UPDATE"]
    nodes = await _nodes(project_id)
    assert nodes["Project.Rooms.r1.Style"].value == "modern"
    assert nodes["Project.Rooms.r1.Style"].changed_by == "user_message"
    assert state1["skipped_rooms"] == [], "a real answer must not trigger the decline/skip mechanic"


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


async def test_complete_context_saves_project():
    async def fake_extract_all(message, known, **kwargs):
        return ExtractedFields(
            projectType="renovation", overallBudget="$20k", timeline="3 months",
            roomType="living room", budgetOrRequirement="$5k-$10k", style="modern",
            squareFootage=300, existingFurniture="none",
            materials=[{"item": "flooring", "material": "oak wood", "specification": "matte finish"}],
        )

    with patch("app.llm.extract_fields", side_effect=fake_extract_all):
        state, _ = await run(base_state("Here's everything about my project", "proj-complete-1"))

    assert state["complete"] is True
    assert state["wrapup_message"] == "wrapup: all set"
    project = await ProjectContext.find_one(ProjectContext.project_id == "proj-complete-1")
    assert project.summary["projectType"] == "renovation"
    assert project.summary["rooms"][0]["roomType"] == "living room"
    assert project.summary["rooms"][0]["materials"][0]["item"] == "flooring"
    assert project.assumptions == []
    node_names = [t.node_name for t in state["trace"]]
    assert "complete_project" in node_names
    # generate_question DOES still run once the project is complete (it's
    # the single consolidated node every branch routes through — see
    # complete_project's own "analyze" edge) but only as a guarded no-op:
    # complete_project already set question_generated=True, so no new
    # question is ever produced.
    assert state["pending_question"] is None


# ---------------------------------------------------------------------------
# Conflict confirmation (new in this cutover)
# ---------------------------------------------------------------------------


async def test_critical_field_restatement_pauses_for_confirmation_not_silent_overwrite():
    project_id = "proj-confirm-flow"

    async def fake_extract_budget_15k(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", overallBudget="$15k")

    with patch("app.llm.extract_fields", side_effect=fake_extract_budget_15k):
        state1, _ = await run(base_state("renovation, budget is $15k", project_id))

    async def fake_extract_budget_20k(message, known, **kwargs):
        return ExtractedFields(overallBudget="$20k")

    state1["message"] = "actually the budget is $20k"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_budget_20k):
        state2, _ = await run(state1)

    assert state2["pending_confirmation"]["old_value"] == "$15k"
    assert state2["pending_confirmation"]["new_value"] == "$20k"
    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "$15k", "must not silently overwrite a critical field"

    state2["message"] = "yes"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract):
        state3, _ = await run(state2)

    assert state3["pending_confirmation"] is None
    nodes3 = await _nodes(project_id)
    assert nodes3["Project.Budget.Total"].value == "$20k", "confirming yes must apply the new value"
    node_names = [t.node_name for t in state3["trace"]]
    assert "confirm_conflict" in node_names


async def test_declining_a_confirmation_keeps_the_old_value():
    project_id = "proj-confirm-decline"

    async def fake_extract_budget_15k(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", overallBudget="$15k")

    with patch("app.llm.extract_fields", side_effect=fake_extract_budget_15k):
        state1, _ = await run(base_state("renovation, budget is $15k", project_id))

    async def fake_extract_budget_20k(message, known, **kwargs):
        return ExtractedFields(overallBudget="$20k")

    state1["message"] = "actually the budget is $20k"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract_budget_20k):
        state2, _ = await run(state1)

    state2["message"] = "no, keep it as is"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with patch("app.llm.extract_fields", side_effect=fake_extract):
        state3, _ = await run(state2)

    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "$15k", "declining must keep the old value"
