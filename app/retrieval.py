"""Scoped Retrieval Engine — Phase 10 (folds in what the plan calls Phase
13 — "never send the whole graph, send relevant paths only" is exactly this
module's job, not separate work).

Replaces "send everything known about the project/active room" (today's
retrieve_context_node, app/graph.py) with "walk to the subtree the query is
actually about and send only that" — using ontology/v1.yaml's canonical
paths, not a growing flat dict.

Room detection is plain fuzzy string matching (rapidfuzz), not embeddings —
a query topic naming a room ("what's the kitchen budget") is a simple
keyword match against a small, already-fetched set of room names; there's
no classification ambiguity here for an embedding search to resolve, unlike
app/canonical_mapper.py's actual job.

Not wired into the live turn — same standing as every module since Phase 1."""

from rapidfuzz import fuzz

from app import graph_store
from app.models import KnowledgeNode

_ROOM_MATCH_THRESHOLD = 70  # rapidfuzz 0-100 scale; same tool app/context_builder.py already uses for dedup

# How many canonical_path segments beyond the resolved root a node may be
# and still be included — e.g. root "Project.Rooms.r1" (3 segments) at
# depth=2 includes "Project.Rooms.r1.Materials.flooring" (5 segments) but
# not ".Materials.flooring.Material" (6 segments, depth 3).
_DEFAULT_DEPTH = 2


async def _existing_rooms(project_id: str) -> dict[str, str]:
    leaves = await graph_store.find_nodes(project_id, node_type="RoomType", lifecycle="active")
    return {leaf.room_id: str(leaf.value) for leaf in leaves if leaf.room_id and leaf.value is not None}


async def resolve_query_to_path(query_topic: str, project_id: str) -> str:
    """"living room budget" -> "Project.Rooms.<room_id>" when a known room is
    named in the query; "Project" (the whole tree, still depth-limited by
    load_subtree) when no room is detected — an ambiguous or project-wide
    query ("what's my overall budget") has no more specific subtree to
    scope to."""
    rooms = await _existing_rooms(project_id)
    best_room_id, best_score = None, 0
    for room_id, room_type in rooms.items():
        score = fuzz.partial_ratio(query_topic.lower(), room_type.lower())
        if score > best_score:
            best_room_id, best_score = room_id, score
    if best_room_id is not None and best_score >= _ROOM_MATCH_THRESHOLD:
        return f"Project.Rooms.{best_room_id}"
    return "Project"


async def load_subtree(project_id: str, root_path: str, depth: int = _DEFAULT_DEPTH) -> list[KnowledgeNode]:
    """Every node at or under root_path, at most `depth` canonical_path
    segments deeper than root_path itself. root_path's own node is included
    (depth 0) when it exists."""
    root_depth = root_path.count(".")
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    return [
        n
        for n in nodes
        if (n.canonical_path == root_path or n.canonical_path.startswith(f"{root_path}."))
        and n.canonical_path.count(".") - root_depth <= depth
    ]


async def retrieve_scoped(query_topic: str, project_id: str, depth: int = _DEFAULT_DEPTH) -> list[KnowledgeNode]:
    target_path = await resolve_query_to_path(query_topic, project_id)
    return await load_subtree(project_id, target_path, depth=depth)
