"""Live-turn coverage for app/graph.py, post-KnowledgeNode-cutover (see
ARCHITECTURE_BASELINE.md and the plan at C:\\Users\\jyoth\\.claude\\plans\\imperative-wibbling-floyd.md).

Anchor-scaffold/connectivity-backstop/candidate-search/revise-retract
behavior from the old ContextGraph era is gone — that machinery doesn't
exist anymore. Equivalent coverage for the pieces that DO still exist
(canonical mapping, alias resolution, dedup) lives in
tests/test_canonical_mapper.py and tests/test_context_builder.py; this file
covers what's specific to the live turn: routing, the question/decline/
confirm flow, and end-to-end completion."""

from unittest.mock import patch

import pytest

from app.deepinfra import ExtractedFields, GraphExtraction
from app.graph import app_graph, classify_intent_node, guard_database_query, guard_split_intents, retrieve_context_node
from app.models import KnowledgeNode, ProjectContext


async def fake_classify(message, history="", pending_field=None, **kwargs):
    lower = message.lower()
    labels: list[str] = []
    if any(k in lower for k in ("budget", "modern", "everything", "cozy", "sofa", "floor", "walnut", "kitchen", "renovation", "$")):
        labels.append("update_context")
    if "recommend" in lower or "products" in lower:
        labels.append("database_query")
    if "remind" in lower or "again" in lower:
        labels.append("context_related")
    if not labels:
        labels.append("direct_question")
    return labels


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


def base_state(message: str, project_id: str = "proj-graph-test", active_room_id: str | None = None) -> dict:
    return {
        "session_id": "test-session",
        "project_id": project_id,
        "message": message,
        "history": "",
        "active_room_id": active_room_id,
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


async def run(state):
    tokens = []
    final_state = None
    async for mode, chunk in app_graph.astream(state, stream_mode=["custom", "values"]):
        if mode == "custom":
            tokens.append(chunk["token"])
        else:
            final_state = chunk
    return final_state, "".join(tokens)


async def _nodes(project_id: str) -> dict[str, KnowledgeNode]:
    return {n.canonical_path: n for n in await KnowledgeNode.find(KnowledgeNode.project_id == project_id).to_list()}


@pytest.fixture(autouse=True)
def mock_models():
    with (
        patch("app.deepinfra.classify_intent", side_effect=fake_classify),
        patch("app.deepinfra.extract_fields", side_effect=fake_extract),
        patch("app.deepinfra.generate_question", side_effect=fake_gen_question),
        patch("app.deepinfra.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.deepinfra.generate_answer", side_effect=fake_answer),
        patch("app.deepinfra.infer_missing_field", side_effect=fake_infer_missing_field),
        patch("app.deepinfra.generate_conflict_confirmation", side_effect=fake_generate_conflict_confirmation),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
    ):
        yield


# ---------------------------------------------------------------------------
# Guards — unchanged behavior, re-exported from app.understanding (Phase 1)
# ---------------------------------------------------------------------------


def test_guard_database_query_downgrades_without_trigger_match():
    assert guard_database_query(["database_query"], "some products for my living room") == ["direct_question"]


def test_guard_split_intents_collapses_plain_joined_fact_to_update_context():
    intents = guard_split_intents(["update_context", "database_query"], "oak flooring for the kitchen, love the modern look")
    assert intents == ["update_context"]


# ---------------------------------------------------------------------------
# Basic routing / structured writes
# ---------------------------------------------------------------------------


async def test_update_context_routes_and_writes_to_knowledge_node():
    state, _ = await run(base_state("I want a modern living room"))
    assert state["intent"] == ["update_context"]
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
    async def fake_classify_with_capture(message, history="", pending_field=None, *, capture=None):
        if capture is not None:
            capture["messages"] = [{"role": "system", "content": "sys"}, {"role": "user", "content": message}]
            capture["raw_output"] = "update_context"
        return ["update_context"]

    with patch("app.deepinfra.classify_intent", side_effect=fake_classify_with_capture):
        result = await classify_intent_node(base_state("modern kitchen"))

    entry = result["trace"][0]
    assert entry.node_name == "classify_intent"
    assert entry.llm_input == [{"role": "system", "content": "sys"}, {"role": "user", "content": "modern kitchen"}]
    assert entry.llm_output == "update_context"


async def test_retrieve_context_node_is_rule_based_with_no_llm_capture():
    result = await retrieve_context_node(base_state("modern kitchen"))
    entry = result["trace"][0]
    assert entry.node_name == "retrieve_context"
    assert entry.llm_input is None
    assert entry.llm_output is None


async def test_extract_fields_failure_does_not_break_turn():
    async def failing_extract(message, known, **kwargs):
        raise Exception("No tool calls or function call found in response (mode: TOOLS)")

    with patch("app.deepinfra.extract_fields", side_effect=failing_extract):
        state, _ = await run(base_state("I want a modern living room"))

    assert state["pending_question"] == "What's your project type?"
    trace_by_node = {t.node_name: t for t in state["trace"]}
    assert "extract_entities failed" in trace_by_node["build_context"].output_summary or "0 node(s)" in trace_by_node["build_context"].output_summary


async def test_direct_question_streams_answer():
    state, tokens = await run(base_state("what colors go with navy blue"))
    assert state["intent"] == ["direct_question"]
    assert state["answer"] == "Sure, here's an answer."
    assert tokens == state["answer"]
    assert state.get("update_summary") is None


# ---------------------------------------------------------------------------
# Split-intent concurrency
# ---------------------------------------------------------------------------


async def test_split_intent_with_update_context_answers_and_writes_together():
    state, _ = await run(base_state("what products would you recommend, and I want a modern living room"))

    assert state["intent"] == ["update_context", "database_query"]
    nodes = await _nodes(state["project_id"])
    assert any(n.node_type == "Style" and n.value == "modern" for n in nodes.values())
    assert state["answer"] == "Sure, here's an answer."
    assert state["pending_question"] == "What's your project type?"

    node_names = [t.node_name for t in state["trace"]]
    assert "query_catalog" in node_names
    assert "build_context" in node_names
    assert "validate_completeness" in node_names
    assert node_names.count("generate_question") == 1
    assert node_names.count("generate_answer") == 1


async def test_split_intent_without_update_context_merges_both_retrievals():
    state, _ = await run(base_state("remind me what you'd recommend and how much they cost"))

    assert state["intent"] == ["database_query", "context_related"]
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
    assert state2["intent"] == ["database_query"]
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

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_flooring):
        state1, _ = await run(base_state("I want oak flooring in the living room", project_id))

    room_id = state1["active_room_id"]
    nodes1 = await _nodes(project_id)
    assert nodes1[f"Project.Rooms.{room_id}.Materials.flooring.Material"].value == "oak wood"

    async def fake_extract_sofa(message, known, **kwargs):
        return ExtractedFields(materials=[{"item": "sofa", "material": "leather"}])

    state1["message"] = "make the sofa leather"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_sofa):
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
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_flooring_update):
        state3, _ = await run(state2)

    nodes3 = await _nodes(project_id)
    assert nodes3[f"Project.Rooms.{room_id}.Materials.flooring.Material"].value == "walnut wood", "same item must be replaced, not duplicated"
    assert nodes3[f"Project.Rooms.{room_id}.Materials.sofa.Material"].value == "leather"


async def test_new_room_mentioned_mid_conversation_does_not_lose_first_room():
    project_id = "proj-new-room-mid"

    async def fake_extract_kitchen(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen", style="modern")

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_kitchen):
        state1, _ = await run(base_state("the kitchen should be modern", project_id))
    kitchen_id = state1["active_room_id"]

    async def fake_extract_bedroom(message, known, **kwargs):
        return ExtractedFields(roomType="bedroom", style="cozy")

    state1["message"] = "the bedroom should be cozy too"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_bedroom):
        state2, _ = await run(state1)
    bedroom_id = state2["active_room_id"]

    assert bedroom_id != kitchen_id
    nodes = await _nodes(project_id)
    assert nodes[f"Project.Rooms.{kitchen_id}.Style"].value == "modern", "first room's data must survive introducing a second room"
    assert nodes[f"Project.Rooms.{bedroom_id}.Style"].value == "cozy"


async def test_typo_respelling_reuses_existing_room_instead_of_forking_a_duplicate():
    project_id = "proj-typo-room"

    async def fake_extract_typo(message, known, **kwargs):
        return ExtractedFields(roomType="bedrroom", style="modern")

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_typo):
        state1, _ = await run(base_state("the bedrroom should be modern", project_id))
    room_id = state1["active_room_id"]

    async def fake_extract_correct_spelling(message, known, **kwargs):
        return ExtractedFields(roomType="bedroom", existingFurniture="none")

    state1["message"] = "add a wardrobe to the bedroom as part of the renovation"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_correct_spelling):
        state2, _ = await run(state1)

    assert state2["active_room_id"] == room_id, "the respelling must resolve to the existing room, not fork a second one"
    nodes = await _nodes(project_id)
    assert nodes[f"Project.Rooms.{room_id}.Style"].value == "modern"
    assert nodes[f"Project.Rooms.{room_id}.ExistingFurniture"].value == "none"


async def test_additional_room_budget_lands_on_the_correct_room_not_the_primary_one():
    project_id = "proj-extra-room-budget"

    async def fake_extract_multi_budget(message, known, **kwargs):
        return ExtractedFields(
            projectType="residential",
            overallBudget="15 lakh",
            roomType="living",
            additionalRoomBudgets=[{"roomType": "kitchen", "budgetOrRequirement": "4 lakh"}],
        )

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_multi_budget):
        state, _ = await run(base_state("living and kitchen, total 15 lakh, kitchen budget 4 lakh", project_id))

    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "15 lakh"
    living_id = state["active_room_id"]
    assert f"Project.Rooms.{living_id}.Budget" not in nodes, "the total must never land on the primary room's own budget field"
    kitchen_budget = next(n for path, n in nodes.items() if n.node_type == "Budget" and n.room_id is not None and n.room_id != living_id)
    assert kitchen_budget.value == "4 lakh"


# ---------------------------------------------------------------------------
# Decline / retry / immediate inference on exhaustion
# ---------------------------------------------------------------------------


def _style_gap_state(project_id: str) -> dict:
    """A state where every field is resolved except the active room's style
    (moderate tier, 1 rephrase allowed)."""
    state = base_state("not sure", project_id)
    state["pending_gap"] = {
        "canonical_path": f"Project.Rooms.r1.Style",
        "field_label": "style",
        "node_type": "Style",
        "room_id": "r1",
        "tier": "moderate",
    }
    state["active_room_id"] = "r1"
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


async def test_decline_rephrases_once_then_infers_immediately_on_exhaustion():
    project_id = "proj-decline-exhaust"
    await _seed_everything_but_style(project_id)
    state = _style_gap_state(project_id)

    state1, _ = await run(state)
    assert state1["pending_question"] == "retry What's your style?"
    assert state1["field_attempts"][f"Project.Rooms.r1.Style"] == 1
    nodes = await _nodes(project_id)
    assert "Project.Rooms.r1.Style" not in nodes, "not yet inferred after just one rephrase"
    node_names = [t.node_name for t in state1["trace"]]
    assert node_names == ["classify_intent", "decline_field", "analyze_context"]

    state1["message"] = "not sure"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False

    state2, _ = await run(state1)
    nodes2 = await _nodes(project_id)
    assert nodes2["Project.Rooms.r1.Style"].changed_by == "inferred"
    assert nodes2["Project.Rooms.r1.Style"].value == "assumed-style"
    node_names2 = [t.node_name for t in state2["trace"]]
    assert "decline_field" in node_names2
    # style is exhausted (moderate tier, limit 1) and immediately inferred,
    # so the next open field (squareFootage) is what gets asked about next.
    assert "square footage" in state2["pending_question"]


async def test_unrelated_answer_is_not_treated_as_a_decline():
    project_id = "proj-decline-unrelated"
    await _seed_everything_but_style(project_id)
    state = _style_gap_state(project_id)
    state["message"] = "modern with a lot of natural light"

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract):
        state1, _ = await run(state)

    assert state1["intent"] == ["update_context"]
    nodes = await _nodes(project_id)
    assert nodes["Project.Rooms.r1.Style"].value == "modern"
    assert nodes["Project.Rooms.r1.Style"].changed_by == "user_message"
    node_names = [t.node_name for t in state1["trace"]]
    assert "decline_field" not in node_names


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

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_all):
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
    assert "generate_question" not in node_names


# ---------------------------------------------------------------------------
# Conflict confirmation (new in this cutover)
# ---------------------------------------------------------------------------


async def test_critical_field_restatement_pauses_for_confirmation_not_silent_overwrite():
    project_id = "proj-confirm-flow"

    async def fake_extract_budget_15k(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", overallBudget="$15k")

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_budget_15k):
        state1, _ = await run(base_state("renovation, budget is $15k", project_id))

    async def fake_extract_budget_20k(message, known, **kwargs):
        return ExtractedFields(overallBudget="$20k")

    state1["message"] = "actually the budget is $20k"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_budget_20k):
        state2, _ = await run(state1)

    assert state2["pending_confirmation"]["old_value"] == "$15k"
    assert state2["pending_confirmation"]["new_value"] == "$20k"
    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "$15k", "must not silently overwrite a critical field"

    state2["message"] = "yes"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract):
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

    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_budget_15k):
        state1, _ = await run(base_state("renovation, budget is $15k", project_id))

    async def fake_extract_budget_20k(message, known, **kwargs):
        return ExtractedFields(overallBudget="$20k")

    state1["message"] = "actually the budget is $20k"
    state1["intent"] = []
    state1["pending_question"] = None
    state1["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract_budget_20k):
        state2, _ = await run(state1)

    state2["message"] = "no, keep it as is"
    state2["intent"] = []
    state2["pending_question"] = None
    state2["question_generated"] = False
    with patch("app.deepinfra.extract_fields", side_effect=fake_extract):
        state3, _ = await run(state2)

    nodes = await _nodes(project_id)
    assert nodes["Project.Budget.Total"].value == "$15k", "declining must keep the old value"
