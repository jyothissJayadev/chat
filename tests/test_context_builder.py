from unittest.mock import patch

import pytest

import app.canonical_mapper as canonical_mapper
from app.canonical_mapper import _FREEFORM_NODE_TYPES, _ONTOLOGY
from app.context_builder import (
    BuildResult,
    Conflict,
    ProposedWrite,
    ResolvedBuild,
    apply_to_graph,
    build_context,
    commit_context,
    detect_conflicts,
    render_project_tree_text,
    resolve_context,
)
from app import graph_store
from app.llm import AdditionalRoomBudget, ExtractedFields, GraphEdgeCreate, GraphExtraction, GraphNodeCreate
from app.models import KnowledgeNode

PROJECT = "proj-cb-1"


@pytest.fixture(autouse=True)
def _reset_type_description_cache():
    canonical_mapper._type_description_cache = None
    yield
    canonical_mapper._type_description_cache = None


async def fake_extract_empty(message, known, **kwargs):
    return ExtractedFields()


async def fake_graph_extraction_empty(message, anchors, candidate_nodes, recent_edges, **kwargs):
    return GraphExtraction(), "clean"


async def fake_embed_routes_to_client_preferences(texts):
    """map_to_canonical's own classification accuracy is covered by
    test_canonical_mapper.py — this fake just needs anything routed through
    it here to land somewhere real (ClientPreferences) so
    context_builder-level tests (dedup, room resolution, conflicts) aren't
    coupled to the mapper's internal scoring."""
    target_desc = _ONTOLOGY["ClientPreferences"]["description"]
    vectors = []
    for text in texts:
        if text == target_desc:
            vectors.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif text in (_ONTOLOGY[t]["description"] for t in _FREEFORM_NODE_TYPES):
            vectors.append([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            vectors.append([1.0, 0.0, 0.0, 0.0, 0.0])
    return vectors


async def _node_at(path: str) -> KnowledgeNode | None:
    return await graph_store.find_one(PROJECT, path)


async def _client_preference_instances(project_id: str) -> list[KnowledgeNode]:
    """ensure_path (canonical_mapper.py) gives the "Project.Requirements.ClientPreferences"
    CONTAINER the same node_type="ClientPreferences" as each actual instance
    under it (same container-vs-instance convention "Rooms" already uses) —
    filter those out to count real preference instances only."""
    nodes = await graph_store.find_nodes(project_id, node_type="ClientPreferences")
    return [n for n in nodes if n.canonical_path != "Project.Requirements.ClientPreferences"]


# ---------------------------------------------------------------------------
# render_project_tree_text — feeds classify_operations' system prompt
# ---------------------------------------------------------------------------


async def test_render_project_tree_text_renders_just_project_for_an_empty_project():
    assert await render_project_tree_text("proj-tree-empty") == "Project"


async def test_render_project_tree_text_matches_the_half_filled_example():
    project_id = "proj-tree-1"
    seed = [
        KnowledgeNode(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", project_id=project_id),
        KnowledgeNode(canonical_path="Project.Budget.Total", node_type="Total", value="$45,000", project_id=project_id),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4", node_type="Rooms", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.RoomType", node_type="RoomType", value="Kitchen", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Budget", node_type="Budget", value="$20,000", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Style", node_type="Style", value="modern farmhouse", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.countertop", node_type="Materials", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.countertop.Label", node_type="Label", value="countertop", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.countertop.Material", node_type="Material", value="quartz", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.countertop.Specification", node_type="Specification", value="white", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.flooring", node_type="Materials", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Materials.flooring.Label", node_type="Label", value="flooring", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Furniture.island", node_type="Furniture", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Furniture.island.Label", node_type="Label", value="island", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.a1b2c3d4.Furniture.island.Notes", node_type="Notes", value="wants seating for 4", project_id=project_id, room_id="a1b2c3d4"),
        KnowledgeNode(canonical_path="Project.Rooms.e5f6g7h8", node_type="Rooms", project_id=project_id, room_id="e5f6g7h8"),
        KnowledgeNode(canonical_path="Project.Rooms.e5f6g7h8.RoomType", node_type="RoomType", value="Bedroom", project_id=project_id, room_id="e5f6g7h8"),
        KnowledgeNode(canonical_path="Project.Requirements.Constraints.no_open_shelving", node_type="Constraints", value=None, project_id=project_id),
        KnowledgeNode(canonical_path="Project.Requirements.Constraints.no_open_shelving.Label", node_type="Label", value="no open shelving", project_id=project_id),
        KnowledgeNode(canonical_path="Project.Requirements.ClientPreferences.warm_neutral_palette", node_type="ClientPreferences", value=None, project_id=project_id),
        KnowledgeNode(canonical_path="Project.Requirements.ClientPreferences.warm_neutral_palette.Label", node_type="Label", value="warm neutral color palette", project_id=project_id),
        KnowledgeNode(canonical_path="Project.Unmapped.pet_friendly_finishes", node_type="Unmapped", value=None, project_id=project_id),
        KnowledgeNode(canonical_path="Project.Unmapped.pet_friendly_finishes.Label", node_type="Label", value="pet-friendly finishes", project_id=project_id),
        # A bare freeform-type container (created lazily by ensure_path, no
        # instance-level Label of its own) must never be mistaken for a real
        # instance — see _is_container.
        KnowledgeNode(canonical_path="Project.Requirements.Constraints", node_type="Constraints", value=None, project_id=project_id),
    ]
    for node in seed:
        await graph_store.insert_node(node)

    text = await render_project_tree_text(project_id)

    assert text == (
        "Project\n"
        '├── BasicInformation\n'
        '│   └── ProjectType = "renovation"\n'
        '├── Budget\n'
        '│   └── Total = "$45,000"\n'
        '├── Rooms\n'
        '│   ├── Rooms.a1b2c3d4\n'
        '│   │   ├── RoomType = "Kitchen"\n'
        '│   │   ├── Budget = "$20,000"\n'
        '│   │   ├── Style = "modern farmhouse"\n'
        '│   │   ├── Furniture.island\n'
        '│   │   │   Label="island", Notes="wants seating for 4"\n'
        '│   │   ├── Materials.countertop\n'
        '│   │   │   Label="countertop", Material="quartz", Specification="white"\n'
        '│   │   └── Materials.flooring\n'
        '│   │       Label="flooring"\n'
        '│   └── Rooms.e5f6g7h8\n'
        '│       └── RoomType = "Bedroom"\n'
        '├── Requirements\n'
        '│   ├── ClientPreferences.warm_neutral_palette\n'
        '│   │   Label="warm neutral color palette"\n'
        '│   └── Constraints.no_open_shelving\n'
        '│       Label="no open shelving"\n'
        '└── Unmapped\n'
        '    └── Unmapped.pet_friendly_finishes\n'
        '        Label="pet-friendly finishes"'
    )
    # No internal bookkeeping ever leaks into the grounding text.
    for forbidden in ("node_id", "confidence", "lifecycle", "tenant_id", "version"):
        assert forbidden not in text
    # SquareFootage/ExistingFurniture/Timeline were never set — omitted
    # entirely rather than shown as blanks.
    assert "SquareFootage" not in text
    assert "ExistingFurniture" not in text
    assert "Timeline" not in text


# ---------------------------------------------------------------------------
# detect_conflicts / apply_to_graph — direct unit tests
# ---------------------------------------------------------------------------


async def test_detect_conflicts_holds_back_critical_tier_value_changes():
    await graph_store.insert_node(KnowledgeNode(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", project_id="proj-conflict-1"))

    proposed = [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical")]
    applyable, conflicts = await detect_conflicts(proposed, "proj-conflict-1")

    assert applyable == []
    assert conflicts == [Conflict(canonical_path="Project.Budget.Total", node_type="Total", old_value="$15k", new_value="$20k", tier="critical")]


async def test_detect_conflicts_auto_applies_moderate_tier_value_changes():
    await graph_store.insert_node(KnowledgeNode(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", project_id="proj-conflict-2", room_id="r1"))

    proposed = [ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="minimalist", room_id="r1", tier="moderate")]
    applyable, conflicts = await detect_conflicts(proposed, "proj-conflict-2")

    assert conflicts == []
    assert applyable == proposed


async def test_detect_conflicts_never_flags_a_brand_new_value():
    proposed = [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical")]
    applyable, conflicts = await detect_conflicts(proposed, "proj-conflict-3")

    assert conflicts == []
    assert applyable == proposed


async def test_apply_to_graph_creates_then_bumps_version_on_update():
    project_id = "proj-apply-1"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", tier="critical")])
    node = await graph_store.find_one(project_id, "Project.Budget.Total")
    assert node.value == "$15k"
    assert node.version == 1

    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical")])
    node = await graph_store.find_one(project_id, "Project.Budget.Total")
    assert node.value == "$20k"
    assert node.version == 2


async def test_apply_to_graph_creates_missing_ancestor_chain():
    project_id = "proj-apply-2"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate")])

    paths = {n.canonical_path for n in await graph_store.find_nodes(project_id)}
    assert {"Project", "Project.Rooms", "Project.Rooms.r1", "Project.Rooms.r1.Style"} <= paths


# ---------------------------------------------------------------------------
# build_context — structured extraction path
# ---------------------------------------------------------------------------


async def test_build_context_writes_structured_fields_to_correct_paths():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", overallBudget="$20k", roomType="kitchen", style="modern", squareFootage=150)

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph_extraction_empty),
    ):
        result = await build_context("modern kitchen, 150 sqft, renovation, 20k budget", PROJECT)

    assert isinstance(result, BuildResult)
    assert (await _node_at("Project.BasicInformation.ProjectType")).value == "renovation"
    assert (await _node_at("Project.Budget.Total")).value == "$20k"
    room_path = f"Project.Rooms.{result.room_id}"
    assert (await _node_at(f"{room_path}.RoomType")).value == "kitchen"
    assert (await _node_at(f"{room_path}.Style")).value == "modern"
    assert (await _node_at(f"{room_path}.SquareFootage")).value == 150


async def test_build_context_reuses_existing_room_via_fuzzy_typo_match():
    project_id = "proj-fuzzy-room"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="bedrroom", room_id="r1", tier="critical")])

    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="bedroom", existingFurniture="none")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph_extraction_empty),
    ):
        result = await build_context("no existing furniture in the bedroom", project_id)

    assert result.room_id == "r1", "the typo'd respelling must resolve to the existing room, not fork a new one"
    rooms = [n for n in await graph_store.find_nodes(project_id) if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms"]
    assert len(rooms) == 1


async def test_build_context_creates_stub_room_for_mentioned_additional_rooms():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="living room", mentionedAdditionalRooms=["kitchen"])

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph_extraction_empty),
    ):
        await build_context("living room and kitchen", "proj-stub-room")

    room_types = {n.value for n in await graph_store.find_nodes("proj-stub-room", node_type="RoomType")}
    assert room_types == {"living room", "kitchen"}


async def test_build_context_routes_additional_room_budget_to_the_correct_room():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(
            roomType="living room",
            additionalRoomBudgets=[AdditionalRoomBudget(roomType="kitchen", budgetOrRequirement="4 lakh")],
        )

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph_extraction_empty),
    ):
        await build_context("living room, and 4 lakh for the kitchen", "proj-extra-budget")

    nodes = await graph_store.find_nodes("proj-extra-budget")
    kitchen_room = next(n for n in nodes if n.node_type == "RoomType" and n.value == "kitchen")
    kitchen_budget = next(n for n in nodes if n.node_type == "Budget" and n.room_id == kitchen_room.room_id)
    assert kitchen_budget.value == "4 lakh"


# ---------------------------------------------------------------------------
# build_context — freeform extraction path (Gap 1 dedup, room-type drop,
# retraction gap)
# ---------------------------------------------------------------------------


async def test_build_context_dedupes_freeform_node_matching_this_turns_structured_style():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="living room", style="modern")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="style_modern", label="Modern style", type="preference")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences) as mock_embed,
    ):
        result = await build_context("modern living room", "proj-dedup-style")

    assert not mock_embed.await_count, "a same-turn duplicate of a structured value must never reach map_to_canonical at all"
    prefs = await _client_preference_instances("proj-dedup-style")
    assert prefs == []
    assert not any("ClientPreferences" in path for path in result.written)


async def test_build_context_routes_genuine_freeform_preference_when_not_a_duplicate():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="living room")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="cozy", label="cozy and warm", type="preference")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences),
    ):
        await build_context("living room, keep it cozy and warm", "proj-genuine-pref")

    prefs = await _client_preference_instances("proj-genuine-pref")
    assert len(prefs) == 1


async def test_build_context_never_routes_a_freeform_room_typed_node_through_the_mapper():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="kitchen_room", label="kitchen", type="room")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences) as mock_embed,
    ):
        await build_context("kitchen", "proj-skip-room-type")

    assert not mock_embed.await_count, "a freeform room-typed proposal must be dropped, never mapped"


async def test_build_context_logs_but_does_not_crash_on_retraction_requests(caplog):
    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(retracted_node_ids=["some_old_node"]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_empty),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
    ):
        result = await build_context("never mind that", "proj-retract")

    assert isinstance(result, BuildResult)
    assert "not applied" in caplog.text


async def test_freeform_relationships_are_reported_but_not_persisted_anywhere():
    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_edges=[GraphEdgeCreate(source="a", target="b", relation="modifies")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract_empty),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
    ):
        result = await build_context("whatever", "proj-edges")

    assert result.freeform_relationships == [{"source": "a", "target": "b", "relation": "modifies"}]


# ---------------------------------------------------------------------------
# Resilience — one failing extraction must not lose the other's success
# ---------------------------------------------------------------------------


async def test_build_context_survives_extract_entities_failure():
    async def failing_extract(message, known, **kwargs):
        raise RuntimeError("boom")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="cozy", label="cozy and warm", type="preference")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=failing_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences),
    ):
        result = await build_context("keep it cozy and warm", "proj-entities-fail")

    assert isinstance(result, BuildResult)
    prefs = await _client_preference_instances("proj-entities-fail")
    assert len(prefs) == 1, "the freeform extraction's success must survive the structured extraction's failure"


async def test_build_context_survives_extract_relationships_failure():
    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(projectType="renovation")

    async def failing_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        raise RuntimeError("boom")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=failing_graph),
    ):
        result = await build_context("residential renovation", "proj-relationships-fail")

    assert isinstance(result, BuildResult)
    assert (await graph_store.find_one("proj-relationships-fail", "Project.BasicInformation.ProjectType")).value == "renovation"
    assert result.freeform_relationships == []


# ---------------------------------------------------------------------------
# resolve_context / commit_context — Step 2 of the classifier-redesign plan
# (see memory/classifier_redesign_decisions.md): build_context split into a
# read-only resolve phase and a write commit phase, so a future clustering
# pass can learn a turn's target paths before anything commits. build_context
# itself is now just resolve_context() + commit_context() — every test above
# already proves that combination behaves correctly; these confirm the split
# itself is sound (resolve alone writes nothing but structured containers,
# commit alone reproduces the exact same graph as build_context).
# ---------------------------------------------------------------------------


async def test_resolve_context_writes_no_field_values_only_gets_previewed():
    project_id = "proj-resolve-no-write"

    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", roomType="kitchen", style="modern")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="cozy", label="cozy and warm", type="preference")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences),
    ):
        resolved = await resolve_context("modern kitchen renovation, keep it cozy and warm", project_id)

    assert isinstance(resolved, ResolvedBuild)
    assert resolved.room_resolution.new_room_type == "kitchen"
    assert len(resolved.freeform_mentions) == 1
    assert resolved.freeform_mentions[0].preview.created_new is True

    # No field VALUE was actually written yet — only resolve_context's own
    # container scaffolding (ensure_path inside map_to_canonical's preview),
    # which never carries a `value`.
    nodes = await graph_store.find_nodes(project_id)
    assert not any(n.value is not None for n in nodes), "resolve_context must not write any field value"
    assert not any(n.node_type == "Label" for n in nodes), "resolve_context must not create the freeform instance/Label leaves"


async def test_commit_context_of_a_resolved_build_matches_build_context_directly():
    """The actual equivalence check: running resolve_context then
    commit_context must produce the identical graph and BuildResult as
    calling build_context() in one shot, for a turn touching structured
    fields, a new room, and a freeform entity all at once."""
    project_id_direct = "proj-equiv-direct"
    project_id_split = "proj-equiv-split"

    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(projectType="renovation", roomType="kitchen", style="modern")

    async def fake_graph(message, anchors, candidate_nodes, recent_edges, **kwargs):
        return GraphExtraction(new_nodes=[GraphNodeCreate(id="cozy", label="cozy and warm", type="preference")]), "clean"

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences),
    ):
        direct_result = await build_context("modern kitchen renovation, keep it cozy and warm", project_id_direct)

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph),
        patch("app.canonical_mapper.llm.embed", side_effect=fake_embed_routes_to_client_preferences),
    ):
        resolved = await resolve_context("modern kitchen renovation, keep it cozy and warm", project_id_split)
        split_result = await commit_context(resolved)

    def _normalize(nodes, room_id):
        # The new room's id is a random uuid4 (see _resolve_room) — different
        # per run by construction, not a real divergence to catch here, so
        # it's replaced with a stable placeholder before comparing.
        return sorted(
            (n.canonical_path.replace(room_id, "<room>"), n.node_type, n.value)
            for n in nodes
        )

    direct_nodes = await graph_store.find_nodes(project_id_direct)
    split_nodes = await graph_store.find_nodes(project_id_split)
    assert _normalize(direct_nodes, direct_result.room_id) == _normalize(split_nodes, split_result.room_id)

    assert direct_result.pending_confirmations == split_result.pending_confirmations
    assert direct_result.freeform_relationships == split_result.freeform_relationships
    assert len(direct_result.written) == len(split_result.written)


async def test_resolved_build_room_resolution_is_none_when_nothing_needs_a_new_room():
    project_id = "proj-resolve-existing-room"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical")])

    async def fake_extract(message, known, **kwargs):
        return ExtractedFields(roomType="kitchen", style="modern")

    with (
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.extract_graph_links", side_effect=fake_graph_extraction_empty),
    ):
        resolved = await resolve_context("modern kitchen", project_id)

    assert resolved.room_resolution.room_id == "r1"
    assert resolved.room_resolution.new_room_type is None, "an existing room match must not be treated as a new one"
