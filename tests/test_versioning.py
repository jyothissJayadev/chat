from unittest.mock import patch

import app.canonical_mapper as canonical_mapper
from app.canonical_mapper import map_to_canonical
from app.context_builder import ProposedWrite, apply_to_graph
from app.models import KnowledgeNode
from app.versioning import get_version_history, record_version

ROOM_ID = "room-ver-1"


async def fake_embed(texts):
    return [[1.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]


def _reset_cache():
    canonical_mapper._type_description_cache = None


# ---------------------------------------------------------------------------
# Exit criterion: "what did the kitchen budget used to be" is answerable by
# a query, not lost.
# ---------------------------------------------------------------------------


async def test_changed_value_is_queryable_via_version_history():
    project_id = "proj-ver-history"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", tier="critical")])
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="moderate")])

    node = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == "Project.Budget.Total")
    history = await get_version_history(node.node_id)

    assert [h.value for h in history] == ["$15k", "$20k"]
    assert [h.version for h in history] == [1, 2]
    assert node.value == "$20k"
    assert node.version == 2


async def test_unchanged_restated_value_produces_no_new_version_row():
    project_id = "proj-ver-noop"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", tier="critical")])
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", tier="critical")])

    node = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == "Project.Budget.Total")
    history = await get_version_history(node.node_id)

    assert node.version == 1, "restating an identical value must not bump version"
    assert len(history) == 1, "a true no-op must not append a redundant history row"


async def test_new_node_creation_produces_its_first_version_row():
    project_id = "proj-ver-create"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate")])

    node = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == "Project.Rooms.r1.Style")
    history = await get_version_history(node.node_id)

    assert len(history) == 1
    assert history[0].value == "modern"
    assert history[0].version == 1
    assert history[0].changed_by == "user_message"


async def test_freeform_node_creation_via_map_to_canonical_is_also_versioned():
    _reset_cache()
    project_id = "proj-ver-freeform"
    with patch("app.canonical_mapper.deepinfra.embed", side_effect=fake_embed):
        match = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)

    label = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == f"{match.canonical_path}.Label")
    history = await get_version_history(label.node_id)

    assert len(history) == 1
    assert history[0].value == "sofa"
    assert history[0].changed_by == "user_message"
    _reset_cache()


async def test_alias_reuse_and_embedding_backfill_do_not_create_spurious_version_rows():
    """Neither appending an alias nor backfilling a missing embedding is a
    `value` change — see app/canonical_mapper.py — so neither should produce
    a version-history entry."""
    _reset_cache()
    project_id = "proj-ver-alias"
    with patch("app.canonical_mapper.deepinfra.embed", side_effect=fake_embed):
        first = await map_to_canonical("sofa", None, project_id, room_id=ROOM_ID)
        await map_to_canonical("couch", None, project_id, room_id=ROOM_ID)  # synonym -> alias reuse, not a new node

    label = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == f"{first.canonical_path}.Label")
    history = await get_version_history(label.node_id)

    assert "couch" in label.aliases
    assert len(history) == 1, "an alias-reuse match must not append a new version row for the same node"
    _reset_cache()


async def test_record_version_captures_changed_by_and_source_message_id():
    node = KnowledgeNode(canonical_path="Project.Budget.Total", node_type="Total", value="$15k", project_id="proj-ver-provenance", version=1)
    await node.insert()

    await record_version(node, changed_by="inferred", source_message_id="msg-123")
    history = await get_version_history(node.node_id)

    assert history[0].changed_by == "inferred"
    assert history[0].source_message_id == "msg-123"
