from app import graph_store
from app.dependency_graph import add_edge, find_dependents, recompute_dependents
from app.models import KnowledgeNode
from app.versioning import get_version_history

PROJECT = "proj-dep-1"


async def _make_node(canonical_path: str, value, project_id: str = PROJECT) -> KnowledgeNode:
    node = KnowledgeNode(canonical_path=canonical_path, node_type="Total", value=value, project_id=project_id)
    return await graph_store.insert_node(node)


async def test_add_edge_is_idempotent():
    project_id = "proj-edge-dedup"
    a, b = await _make_node("Project.Budget.Total", "$20k", project_id), await _make_node("Project.Rooms.r1.Budget", "$8k", project_id)

    edge1 = await add_edge(b.node_id, a.node_id, "derives_from", project_id)
    edge2 = await add_edge(b.node_id, a.node_id, "derives_from", project_id)

    assert edge1.edge_id == edge2.edge_id
    all_edges = await graph_store.find_edges(project_id)
    assert len(all_edges) == 1


async def test_find_dependents_returns_sources_targeting_the_given_node():
    project_id = "proj-find-deps"
    kitchen_budget = await _make_node("Project.Rooms.r1.Budget", "$8k", project_id)
    cabinet_budget = await _make_node("Project.Rooms.r1.Furniture.cabinet.Budget", "$1.2k", project_id)
    unrelated = await _make_node("Project.Rooms.r2.Budget", "$5k", project_id)

    await add_edge(cabinet_budget.node_id, kitchen_budget.node_id, "derives_from", project_id)

    dependents = await find_dependents(kitchen_budget.node_id, project_id)
    dependent_ids = {n.node_id for n in dependents}

    assert dependent_ids == {cabinet_budget.node_id}
    assert unrelated.node_id not in dependent_ids


async def test_find_dependents_filters_by_relation_when_given():
    project_id = "proj-find-deps-relation"
    room = await _make_node("Project.Rooms.r1.Budget", "$8k", project_id)
    derived = await _make_node("Project.Rooms.r1.Furniture.cabinet.Budget", "$1.2k", project_id)
    merely_related = await _make_node("Project.Rooms.r1.Furniture.cabinet.Notes", "matches room budget", project_id)

    await add_edge(derived.node_id, room.node_id, "derives_from", project_id)
    await add_edge(merely_related.node_id, room.node_id, "modifies", project_id)

    only_derived = await find_dependents(room.node_id, project_id, relation="derives_from")
    assert {n.node_id for n in only_derived} == {derived.node_id}

    everything = await find_dependents(room.node_id, project_id)
    assert {n.node_id for n in everything} == {derived.node_id, merely_related.node_id}


async def test_find_dependents_returns_empty_list_for_a_node_with_no_edges():
    project_id = "proj-no-deps"
    lonely = await _make_node("Project.Budget.Total", "$20k", project_id)
    assert await find_dependents(lonely.node_id, project_id) == []


# ---------------------------------------------------------------------------
# recompute_dependents — the mechanism, exercised with a worked example (a
# simple "N% of parent" rule, defined here in the test, NOT in
# app/dependency_graph.py — see that module's docstring for why).
# ---------------------------------------------------------------------------


async def _fifteen_percent_of_parent(dependent: KnowledgeNode, changed: KnowledgeNode):
    if not isinstance(changed.value, (int, float)):
        return None
    return round(changed.value * 0.15, 2)


async def test_recompute_dependents_cascades_a_value_change_to_a_dependent():
    """Exit criterion: changing a room budget produces a correct cascading
    update to at least one dependent (cabinet budget), recorded through the
    normal versioning path."""
    project_id = "proj-cascade"
    kitchen_budget = await _make_node("Project.Rooms.r1.Budget", 20000, project_id)
    cabinet_budget = await _make_node("Project.Rooms.r1.Furniture.cabinet.Budget", 1000, project_id)
    await add_edge(cabinet_budget.node_id, kitchen_budget.node_id, "derives_from", project_id)

    kitchen_budget.value = 30000
    kitchen_budget.version += 1
    await graph_store.save_node(kitchen_budget)

    updated = await recompute_dependents(kitchen_budget, project_id, _fifteen_percent_of_parent)

    assert len(updated) == 1
    assert updated[0].node_id == cabinet_budget.node_id
    assert updated[0].value == 4500.0

    refreshed = await graph_store.find_by_node_id(cabinet_budget.node_id)
    assert refreshed.value == 4500.0
    assert refreshed.version == 2

    # _make_node inserts directly (bypassing record_version), so the node's
    # initial value never got a history row — only the cascaded update did.
    history = await get_version_history(cabinet_budget.node_id)
    assert [h.value for h in history] == [4500.0]
    assert history[-1].changed_by == "system_default", "a deterministic cascaded recalculation is system_default, not user_message or inferred (an LLM guess)"


async def test_recompute_dependents_skips_a_dependent_the_rule_declines_to_touch():
    project_id = "proj-cascade-skip"
    non_numeric_budget = await _make_node("Project.Rooms.r1.Budget", "roughly twenty thousand", project_id)
    cabinet_budget = await _make_node("Project.Rooms.r1.Furniture.cabinet.Budget", 1000, project_id)
    await add_edge(cabinet_budget.node_id, non_numeric_budget.node_id, "derives_from", project_id)

    updated = await recompute_dependents(non_numeric_budget, project_id, _fifteen_percent_of_parent)

    assert updated == []
    unchanged = await graph_store.find_by_node_id(cabinet_budget.node_id)
    assert unchanged.value == 1000
    assert unchanged.version == 1


async def test_recompute_dependents_skips_a_result_identical_to_the_current_value():
    project_id = "proj-cascade-noop"
    kitchen_budget = await _make_node("Project.Rooms.r1.Budget", 20000, project_id)
    cabinet_budget = await _make_node("Project.Rooms.r1.Furniture.cabinet.Budget", 3000.0, project_id)
    await add_edge(cabinet_budget.node_id, kitchen_budget.node_id, "derives_from", project_id)

    updated = await recompute_dependents(kitchen_budget, project_id, _fifteen_percent_of_parent)

    assert updated == []
    history = await get_version_history(cabinet_budget.node_id)
    assert history == [], "a recomputed value identical to the current one must not append a spurious version row"
