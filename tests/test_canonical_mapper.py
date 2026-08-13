"""Golden-set validation for app.canonical_mapper — see Phase 5's exit
criteria: the mapper must resolve a hand-built synonym set correctly, log
(not silently drop) low-confidence creations, and never produce a
canonical_path outside ontology/v1.yaml.

llm.embed is mocked with a small deterministic vector scheme (not real
embeddings — there's no live model call in tests) built specifically so the
REAL cosine-similarity/thresholding logic in canonical_mapper.py gets
exercised, not just hand-asserted: every phrase's vector is
[0.8 * type-one-hot, 0.6 * entity-one-hot] (unit norm). This guarantees, by
construction: (a) synonyms of the SAME entity are identical vectors (cosine
1.0 — always resolve together), (b) different entities of the SAME ontology
type share only their type component (cosine 0.8^2 = 0.64 < the mapper's
0.75 threshold — must NOT merge), and (c) a phrase against its own type's
pure-type-axis ontology description scores 0.8 (>= threshold — type
inference succeeds). See the 0.75 <= w1 < 0.866 derivation this depends on
if the threshold in app/canonical_mapper.py ever changes.
"""

from unittest.mock import patch

import pytest

import app.canonical_mapper as canonical_mapper
from app import graph_store
from app.canonical_mapper import _FREEFORM_NODE_TYPES, _ONTOLOGY, map_to_canonical

_TYPE_AXIS_WEIGHT = 0.8
_ENTITY_AXIS_WEIGHT = 0.6
_ROOM_SCOPED_TYPES = ("Materials", "Furniture", "Attributes")
ROOM_ID = "room-abc123"


@pytest.fixture(autouse=True)
def _reset_type_description_cache():
    """The module-level ontology-type-description embedding cache (Phase 6)
    persists for the process lifetime by design — reset it around every test
    so embed-call-count assertions don't depend on what earlier tests in this
    file happened to warm it with."""
    canonical_mapper._type_description_cache = None
    yield
    canonical_mapper._type_description_cache = None

# (entity_id, node_type, [synonym phrases...]) — hand-built, interior-design
# specific, covering all five freeform ontology types with at least two
# distinct entities per type (so same-type-but-different-entity non-merging
# is actually exercised, not just same-entity merging).
_ENTITIES: list[tuple[str, str, list[str]]] = [
    ("tv_unit", "Furniture", ["TV unit", "TV cabinet", "entertainment unit", "TV console"]),
    ("sofa", "Furniture", ["sofa", "couch", "settee"]),
    ("wardrobe", "Furniture", ["wardrobe", "closet", "almirah"]),
    ("chandelier", "Furniture", ["chandelier", "pendant light", "hanging light fixture"]),
    ("quartz_countertop", "Materials", ["quartz countertop", "quartz counter", "quartz worktop"]),
    ("oak_flooring", "Materials", ["oak flooring", "oak wood flooring", "oak floor"]),
    ("accent_wall", "Attributes", ["accent wall", "feature wall", "statement wall"]),
    ("lacquer_finish", "Attributes", ["matte lacquer finish", "matte finish lacquer", "lacquered matte finish"]),
    ("budget_cap", "Constraints", ["must not exceed 5 lakh", "keep it under 5 lakh", "budget cannot go above 5 lakh"]),
    ("avoid_dark", "Constraints", ["avoid dark colors", "no dark tones anywhere", "steer clear of dark shades"]),
    ("cozy_mood", "ClientPreferences", ["cozy and warm feel", "warm cozy atmosphere", "inviting warm vibe"]),
]

_NUM_TYPES = len(_FREEFORM_NODE_TYPES)
_ENTITY_INDEX = {entity_id: i for i, (entity_id, _, _) in enumerate(_ENTITIES)}
_NUM_ENTITIES = len(_ENTITY_INDEX)


def _type_axis(node_type: str) -> list[float]:
    v = [0.0] * _NUM_TYPES
    v[_FREEFORM_NODE_TYPES.index(node_type)] = 1.0
    return v


def _entity_axis(entity_id: str) -> list[float]:
    v = [0.0] * _NUM_ENTITIES
    v[_ENTITY_INDEX[entity_id]] = 1.0
    return v


def _phrase_vector(node_type: str, entity_id: str) -> list[float]:
    return [_TYPE_AXIS_WEIGHT * x for x in _type_axis(node_type)] + [
        _ENTITY_AXIS_WEIGHT * x for x in _entity_axis(entity_id)
    ]


_PHRASE_TO_VECTOR: dict[str, list[float]] = {}
for _entity_id, _node_type, _phrases in _ENTITIES:
    _vec = _phrase_vector(_node_type, _entity_id)
    for _phrase in _phrases:
        _PHRASE_TO_VECTOR[_phrase.lower()] = _vec

_LOW_CONFIDENCE_PHRASE = "something completely ambiguous"
_PHRASE_TO_VECTOR[_LOW_CONFIDENCE_PHRASE] = [0.0] * (_NUM_TYPES + _NUM_ENTITIES)

_DESCRIPTION_TO_VECTOR = {
    _ONTOLOGY[t]["description"]: _type_axis(t) + [0.0] * _NUM_ENTITIES for t in _FREEFORM_NODE_TYPES
}


async def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        if text in _DESCRIPTION_TO_VECTOR:
            vectors.append(_DESCRIPTION_TO_VECTOR[text])
        elif text.lower() in _PHRASE_TO_VECTOR:
            vectors.append(_PHRASE_TO_VECTOR[text.lower()])
        else:
            raise AssertionError(f"fake_embed received unexpected text: {text!r} — golden set is missing this phrase")
    return vectors


def _room_for(node_type: str) -> str | None:
    return ROOM_ID if node_type in _ROOM_SCOPED_TYPES else None


@pytest.mark.parametrize("entity_id,node_type,phrases", _ENTITIES)
async def test_golden_set_synonyms_resolve_to_the_same_node(entity_id, node_type, phrases):
    project_id = f"proj-golden-{entity_id}"
    room_id = _room_for(node_type)

    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        first = await map_to_canonical(phrases[0], None, project_id, room_id=room_id)
        assert first.node_type == node_type, f"{phrases[0]!r} should infer node_type={node_type}"
        assert first.created_new is True

        for phrase in phrases[1:]:
            match = await map_to_canonical(phrase, None, project_id, room_id=room_id)
            assert match.node_id == first.node_id, f"{phrase!r} must resolve to the same node as {phrases[0]!r}"
            assert match.created_new is False


async def test_different_entities_of_the_same_type_do_not_merge():
    project_id = "proj-golden-distinct-furniture"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        sofa = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        wardrobe = await map_to_canonical("wardrobe", None, project_id, room_id=ROOM_ID)

    assert sofa.node_type == wardrobe.node_type == "Furniture"
    assert sofa.node_id != wardrobe.node_id, "distinct furniture pieces of the same type must not collapse into one node"


async def test_exact_rematch_short_circuits_before_any_embedding_call():
    project_id = "proj-exact-rematch"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed) as mock_embed:
        first = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        # 2 calls for a brand-new node with no existing candidates: (1) the
        # query text itself, (2) the 5 type descriptions, embedded once and
        # cached module-wide (see _reset_type_description_cache) — not
        # re-embedded on a later call even for a different project/entity.
        assert mock_embed.await_count == 2

        second = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)

    assert second.matched_via == "alias_exact"
    assert second.node_id == first.node_id
    assert mock_embed.await_count == 2, "an exact re-mention must not trigger another embedding call"


async def test_existing_candidate_embedding_is_persisted_and_reused_not_recomputed():
    """Phase 6: a candidate's embedding, once computed, is stored on its
    Label node and reused on later calls instead of being re-embedded every
    time — only the NEW query text costs an embedding call on a repeat
    lookup against an already-embedded candidate pool."""
    project_id = "proj-embedding-reuse"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed) as mock_embed:
        first = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        calls_after_first = mock_embed.await_count

        label = await graph_store.find_one(project_id, f"{first.canonical_path}.Label")
        assert label.embedding is not None, "a newly-created node's embedding must be persisted, not left unset"

        # "couch" is a synonym of "sofa" — this call must score against
        # sofa's Label using its PERSISTED embedding, costing exactly one
        # more embed() call (for "couch" itself), not two.
        second = await map_to_canonical("couch", None, project_id, room_id=ROOM_ID)

    assert second.node_id == first.node_id
    assert second.matched_via == "alias_embedding"
    assert mock_embed.await_count == calls_after_first + 1, (
        "the existing candidate's already-persisted embedding must not be recomputed"
    )


async def test_node_type_hint_skips_type_inference_embedding():
    project_id = "proj-hint-test"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        # Deliberately NOT in the golden set — would raise in fake_embed if
        # type-inference embedding were attempted despite the hint. No
        # existing candidates for this fresh project either, so the
        # alias-embedding step also never calls embed().
        match = await map_to_canonical("some brand new gizmo", "Furniture", project_id, room_id=ROOM_ID)

    assert match.node_type == "Furniture"
    assert match.matched_via == "type_hint"
    assert match.confidence == 1.0
    assert match.created_new is True


async def test_low_confidence_mention_falls_back_to_unmapped_and_is_flagged():
    project_id = "proj-unmapped-test"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        match = await map_to_canonical(_LOW_CONFIDENCE_PHRASE, None, project_id, room_id=ROOM_ID)

    assert match.node_type == "Unmapped"
    assert match.matched_via == "unmapped"
    assert match.flagged_for_review is True
    assert match.canonical_path == f"Project.Rooms.{ROOM_ID}.Unmapped.something_completely_ambiguous"


async def test_room_scoped_hint_without_room_id_falls_back_to_project_level_unmapped():
    """Gap 5's resolution: a room-scoped type with no room context can't be
    placed under Rooms.<instance>.Furniture — it falls back to the
    project-level Unmapped bucket instead of being silently dropped or
    guessed into the wrong room."""
    project_id = "proj-unmapped-no-room"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        match = await map_to_canonical("some furniture thing", "Furniture", project_id, room_id=None)

    assert match.node_type == "Unmapped"
    assert match.canonical_path == "Project.Unmapped.some_furniture_thing"
    assert match.flagged_for_review is True


def _all_valid_node_types() -> set[str]:
    """A node_type is valid if it's either a top-level ontology/v1.yaml key
    (a container/instantiable type, e.g. "Furniture") or a leaf field name
    listed in some type's `fields:` (e.g. "Label", "Material" — these are
    node_type values Phase 4/5 both use for leaves, never top-level keys
    themselves)."""
    top_level = {k for k in _ONTOLOGY if k not in ("version", "versioning_policy")}
    leaf_fields = {f for spec in _ONTOLOGY.values() if isinstance(spec, dict) for f in spec.get("fields", [])}
    return top_level | leaf_fields


async def test_never_produces_a_node_type_outside_the_ontology():
    project_id = "proj-ontology-fidelity"
    valid_node_types = _all_valid_node_types()

    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        await map_to_canonical("quartz countertop", None, project_id, room_id=ROOM_ID)
        await map_to_canonical(_LOW_CONFIDENCE_PHRASE, None, project_id, room_id=ROOM_ID)

    nodes = await graph_store.find_nodes(project_id)
    assert len(nodes) > 3, "expected container-chain nodes plus each instance and its Label leaf"
    for node in nodes:
        assert node.node_type in valid_node_types, f"{node.node_type!r} is not a real ontology/v1.yaml node type or field"


async def test_created_instance_has_a_findable_label_leaf():
    project_id = "proj-label-leaf"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        match = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)

    label = await graph_store.find_one(project_id, f"{match.canonical_path}.Label")
    assert label is not None
    assert label.value == "sofa"
    assert label.parent_id == match.node_id


# ---------------------------------------------------------------------------
# commit=False — the resolve-only preview app.context_builder.resolve_context
# uses (Step 2 of the classifier-redesign plan). A preview must describe
# exactly what a real commit=True call would do, without actually creating
# anything, so a caller can safely use it purely for clustering decisions.
# ---------------------------------------------------------------------------


async def test_preview_of_a_new_entity_creates_nothing():
    project_id = "proj-preview-no-write"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        preview = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID, commit=False)

    assert preview.created_new is True
    assert preview.node_id == "", "nothing was actually created, so there's no real node id yet"
    assert preview.canonical_path == f"Project.Rooms.{ROOM_ID}.Furniture.sofa"
    nodes = await graph_store.find_nodes(project_id)
    assert not any(n.node_type == "Label" for n in nodes), "commit=False must not create the instance/Label leaves"


async def test_preview_path_matches_what_a_later_commit_actually_creates():
    project_id = "proj-preview-matches-commit"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        preview = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID, commit=False)
        committed = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID, commit=True)

    assert committed.canonical_path == preview.canonical_path
    assert committed.node_type == preview.node_type
    assert committed.created_new is True


async def test_preview_of_an_already_existing_entity_matches_without_creating_a_duplicate():
    project_id = "proj-preview-existing"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        first = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        preview = await map_to_canonical("couch", None, project_id, room_id=ROOM_ID, commit=False)

    assert preview.created_new is False
    assert preview.matched_via == "alias_embedding"
    assert preview.node_id == first.node_id
    assert preview.canonical_path == first.canonical_path


async def test_preview_of_a_low_confidence_mention_previews_the_unmapped_path():
    project_id = "proj-preview-unmapped"
    with patch("app.canonical_mapper.llm.embed", side_effect=fake_embed):
        preview = await map_to_canonical(_LOW_CONFIDENCE_PHRASE, None, project_id, room_id=ROOM_ID, commit=False)

    assert preview.node_type == "Unmapped"
    assert preview.matched_via == "unmapped"
    assert preview.flagged_for_review is True
    assert preview.canonical_path == f"Project.Rooms.{ROOM_ID}.Unmapped.something_completely_ambiguous"
    nodes = await graph_store.find_nodes(project_id)
    assert not any(n.node_type == "Label" for n in nodes)


# ---------------------------------------------------------------------------
# is_grounded_connection / room_id_from_connection / split_connection —
# app.llm.Operation.connection's shape reasoning, shared by
# app.execution's write clustering and app.graph's delete-target resolution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection,expected",
    [
        ("Rooms.a1b2c3d4", True),
        ("Rooms.a1b2c3d4.Materials.countertop", True),
        ("BasicInformation.ProjectType", True),
        ("Project", True),
        ("Living Room", False),
        ("Kids Bedroom", False),
    ],
)
def test_is_grounded_connection(connection, expected):
    assert canonical_mapper.is_grounded_connection(connection) is expected


@pytest.mark.parametrize(
    "connection,expected_room_id",
    [
        ("Rooms.a1b2c3d4", "a1b2c3d4"),
        ("Rooms.a1b2c3d4.Materials.countertop", "a1b2c3d4"),
        ("Project", None),
        ("Budget.Total", None),
        ("Living Room", None),
    ],
)
def test_room_id_from_connection(connection, expected_room_id):
    assert canonical_mapper.room_id_from_connection(connection) == expected_room_id


def test_split_connection_grounded_room_path_yields_room_id_override_only():
    room_id, room_hint = canonical_mapper.split_connection("Rooms.a1b2c3d4.Furniture.island")
    assert room_id == "a1b2c3d4"
    assert room_hint is None


def test_split_connection_free_text_yields_room_hint_override_only():
    room_id, room_hint = canonical_mapper.split_connection("Living Room")
    assert room_id is None
    assert room_hint == "Living Room"


def test_split_connection_none_yields_no_overrides():
    assert canonical_mapper.split_connection(None) == (None, None)
