from app import graph_store
from app.context_builder import ProposedWrite, apply_to_graph
from app.retrieval import load_subtree, resolve_query_to_path, retrieve_scoped


async def _seed_multi_room_project(project_id: str) -> dict[str, str]:
    """Three rooms, each with a budget, style, and a couple of materials —
    enough structure that scoping to one room actually excludes a
    meaningful amount of the other two."""
    rooms = {}
    for room_id, room_type, budget, style in [
        ("kitchen1", "kitchen", "$8k", "modern"),
        ("bedroom1", "bedroom", "$5k", "cozy"),
        ("living1", "living room", "$10k", "minimalist"),
    ]:
        rooms[room_id] = room_type
        await apply_to_graph(
            project_id,
            [
                ProposedWrite(canonical_path=f"Project.Rooms.{room_id}.RoomType", node_type="RoomType", value=room_type, room_id=room_id, tier="critical"),
                ProposedWrite(canonical_path=f"Project.Rooms.{room_id}.Budget", node_type="Budget", value=budget, room_id=room_id, tier="critical"),
                ProposedWrite(canonical_path=f"Project.Rooms.{room_id}.Style", node_type="Style", value=style, room_id=room_id, tier="moderate"),
                ProposedWrite(canonical_path=f"Project.Rooms.{room_id}.Materials.flooring.Label", node_type="Label", value="flooring", room_id=room_id, tier="optional"),
                ProposedWrite(canonical_path=f"Project.Rooms.{room_id}.Materials.flooring.Material", node_type="Material", value="oak wood", room_id=room_id, tier="optional"),
            ],
        )
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$23k", tier="critical")])
    return rooms


async def test_resolve_query_to_path_matches_a_named_room():
    project_id = "proj-retrieval-resolve"
    await _seed_multi_room_project(project_id)
    assert await resolve_query_to_path("what's the kitchen budget", project_id) == "Project.Rooms.kitchen1"


async def test_resolve_query_to_path_falls_back_to_project_root_when_no_room_named():
    project_id = "proj-retrieval-fallback"
    await _seed_multi_room_project(project_id)
    assert await resolve_query_to_path("what's my overall budget", project_id) == "Project"


async def test_load_subtree_respects_depth_limit():
    project_id = "proj-retrieval-depth"
    await _seed_multi_room_project(project_id)

    shallow = await load_subtree(project_id, "Project.Rooms.kitchen1", depth=2)
    shallow_paths = {n.canonical_path for n in shallow}
    assert "Project.Rooms.kitchen1.Style" in shallow_paths
    assert "Project.Rooms.kitchen1.Materials.flooring" in shallow_paths
    assert "Project.Rooms.kitchen1.Materials.flooring.Material" not in shallow_paths, "one level too deep for depth=2"

    deeper = await load_subtree(project_id, "Project.Rooms.kitchen1", depth=3)
    assert "Project.Rooms.kitchen1.Materials.flooring.Material" in {n.canonical_path for n in deeper}


async def test_retrieve_scoped_excludes_other_rooms_but_keeps_the_answer():
    project_id = "proj-retrieval-scoped"
    await _seed_multi_room_project(project_id)

    scoped = await retrieve_scoped("what's the kitchen budget", project_id)
    scoped_paths = {n.canonical_path for n in scoped}

    assert "Project.Rooms.kitchen1.Budget" in scoped_paths, "the actual answer must survive scoping"
    assert not any(p.startswith("Project.Rooms.bedroom1") for p in scoped_paths)
    assert not any(p.startswith("Project.Rooms.living1") for p in scoped_paths)


async def test_retrieve_scoped_sends_meaningfully_fewer_tokens_than_the_whole_project():
    """Exit criterion: token count sent to generate_answer drops measurably
    for a context-related query. Approximated here as character count of the
    serialized node set, same proxy app/graph.py's own retrieve_context_node
    already uses (a joined summary string, not an actual tokenizer call)."""
    project_id = "proj-retrieval-tokens"
    await _seed_multi_room_project(project_id)

    whole_project = await graph_store.find_nodes(project_id)
    scoped = await retrieve_scoped("what's the kitchen budget", project_id)

    whole_chars = sum(len(str(n.value)) + len(n.canonical_path) for n in whole_project)
    scoped_chars = sum(len(str(n.value)) + len(n.canonical_path) for n in scoped)

    assert scoped_chars < whole_chars
    assert scoped_chars <= whole_chars * 0.6, "scoping to one of three rooms should cut the payload substantially, not marginally"
