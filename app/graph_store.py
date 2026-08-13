"""Cypher CRUD layer for the knowledge graph — the Neo4j equivalent of what
Beanie gave app/context_builder.py, app/canonical_mapper.py, and
app/versioning.py directly against KnowledgeNode as a Mongo Document. Every
other module that used to write `KnowledgeNode.find(...)`/`.insert()`/
`.save()` now calls a function here instead — the query layer changed, the
call sites' logic didn't.

Graph shape (see the Neo4j migration plan):
  (:KNode {node_id, canonical_path, node_type, value, ...})
    -[:CHILD_OF]->(:KNode)              parent_id, but a real edge
    -[:REL {relation, edge_id, project_id, created_at}]->(:KNode)   KnowledgeEdge

`:REL` is one generic relationship type carrying `relation` as a property,
not one Neo4j relationship type per KnowledgeEdgeRelation value — Cypher
can't parameterize a relationship TYPE (only property values), so the
alternative is building relationship-type Cypher via string interpolation,
which is worse than a property filter for a value that's already a closed
Literal enum on the Python side.

No relationship links a :KNodeVersion to its :KNode — every read is by
`node_id` property (indexed, app/neo4j_db.py), so a graph edge wouldn't
speed up anything, only add a write to every version insert."""

from typing import Any, Optional

from neo4j.time import DateTime as Neo4jDateTime

from app import neo4j_db
from app.models import KnowledgeEdge, KnowledgeEdgeRelation, KnowledgeNode, KnowledgeNodeVersion

_DATETIME_FIELDS = ("created_at", "updated_at", "changed_at")


def _to_native(value: Any) -> Any:
    return value.to_native() if isinstance(value, Neo4jDateTime) else value


def _node_from_props(props: dict) -> KnowledgeNode:
    data = dict(props)
    data["parent_id"] = data.pop("_parent_id", None)
    for field in _DATETIME_FIELDS:
        if field in data:
            data[field] = _to_native(data[field])
    return KnowledgeNode(**data)


def _version_from_props(props: dict) -> KnowledgeNodeVersion:
    data = dict(props)
    for field in _DATETIME_FIELDS:
        if field in data:
            data[field] = _to_native(data[field])
    return KnowledgeNodeVersion(**data)


def _edge_from_props(props: dict) -> KnowledgeEdge:
    data = dict(props)
    for field in _DATETIME_FIELDS:
        if field in data:
            data[field] = _to_native(data[field])
    return KnowledgeEdge(**data)


# ---------------------------------------------------------------------------
# KnowledgeNode
# ---------------------------------------------------------------------------


async def find_nodes(
    project_id: str,
    *,
    node_type: Optional[str] = None,
    lifecycle: Optional[str] = None,
    changed_by: Optional[str] = None,
) -> list[KnowledgeNode]:
    cypher = """
    MATCH (n:KNode {project_id: $project_id})
    WHERE ($node_type IS NULL OR n.node_type = $node_type)
      AND ($lifecycle IS NULL OR n.lifecycle = $lifecycle)
      AND ($changed_by IS NULL OR n.changed_by = $changed_by)
    OPTIONAL MATCH (n)-[:CHILD_OF]->(p:KNode)
    RETURN n{.*, _parent_id: p.node_id} AS node
    """
    async with neo4j_db.session() as session:
        result = await session.run(
            cypher, project_id=project_id, node_type=node_type, lifecycle=lifecycle, changed_by=changed_by
        )
        records = [r["node"] async for r in result]

    return [_node_from_props(r) for r in records]


async def find_one(project_id: str, canonical_path: str) -> Optional[KnowledgeNode]:
    cypher = """
    MATCH (n:KNode {project_id: $project_id, canonical_path: $canonical_path})
    OPTIONAL MATCH (n)-[:CHILD_OF]->(p:KNode)
    RETURN n{.*, _parent_id: p.node_id} AS node
    LIMIT 1
    """
    async with neo4j_db.session() as session:
        result = await session.run(cypher, project_id=project_id, canonical_path=canonical_path)
        record = await result.single()
    return _node_from_props(record["node"]) if record else None


async def find_by_node_id(node_id: str) -> Optional[KnowledgeNode]:
    cypher = """
    MATCH (n:KNode {node_id: $node_id})
    OPTIONAL MATCH (n)-[:CHILD_OF]->(p:KNode)
    RETURN n{.*, _parent_id: p.node_id} AS node
    LIMIT 1
    """
    async with neo4j_db.session() as session:
        result = await session.run(cypher, node_id=node_id)
        record = await result.single()
    return _node_from_props(record["node"]) if record else None


async def insert_node(node: KnowledgeNode) -> KnowledgeNode:
    """Creates `node` and, if it has a parent_id, links it under that parent
    via :CHILD_OF — both in one write transaction, so a node is never left
    half-attached."""
    props = node.model_dump(exclude={"parent_id"})

    async def work(tx):
        await tx.run("CREATE (n:KNode) SET n = $props", props=props)
        if node.parent_id:
            await tx.run(
                "MATCH (n:KNode {node_id: $node_id}), (p:KNode {node_id: $parent_id}) CREATE (n)-[:CHILD_OF]->(p)",
                node_id=node.node_id,
                parent_id=node.parent_id,
            )

    async with neo4j_db.session() as session:
        await session.execute_write(work)
    return node


async def save_node(node: KnowledgeNode) -> None:
    """Full property overwrite by node_id — mirrors Beanie's `.save()`
    (whatever the model no longer carries is dropped: `SET n = $props`
    clears any property missing from the map, it doesn't merge). Never
    touches :CHILD_OF — a node's parent is set once, at creation."""
    props = node.model_dump(exclude={"parent_id"})
    async with neo4j_db.session() as session:
        await session.run("MATCH (n:KNode {node_id: $node_id}) SET n = $props", node_id=node.node_id, props=props)


async def descendant_ids(root_node_id: str) -> list[str]:
    """root_node_id and every node reachable by walking :CHILD_OF backwards
    (i.e. root plus all its descendants) — the traversal
    versioning.retract_subtree needs, done as one graph query instead of a
    breadth-first walk in Python re-fetching each generation."""
    cypher = """
    MATCH (root:KNode {node_id: $root_id})
    OPTIONAL MATCH (root)<-[:CHILD_OF*0..]-(descendant:KNode)
    RETURN COLLECT(DISTINCT descendant.node_id) AS ids
    """
    async with neo4j_db.session() as session:
        result = await session.run(cypher, root_id=root_node_id)
        record = await result.single()
    return record["ids"] if record else []


# ---------------------------------------------------------------------------
# KnowledgeNodeVersion
# ---------------------------------------------------------------------------


async def insert_version(version: KnowledgeNodeVersion) -> None:
    props = version.model_dump()
    async with neo4j_db.session() as session:
        await session.run("CREATE (v:KNodeVersion) SET v = $props", props=props)


async def find_versions(node_id: str) -> list[KnowledgeNodeVersion]:
    cypher = "MATCH (v:KNodeVersion {node_id: $node_id}) RETURN v{.*} AS version ORDER BY v.version"
    async with neo4j_db.session() as session:
        result = await session.run(cypher, node_id=node_id)
        records = [r["version"] async for r in result]
    return [_version_from_props(r) for r in records]


# ---------------------------------------------------------------------------
# KnowledgeEdge
# ---------------------------------------------------------------------------


async def find_edge(project_id: str, source_id: str, target_id: str, relation: KnowledgeEdgeRelation) -> Optional[KnowledgeEdge]:
    cypher = """
    MATCH (s:KNode {node_id: $source_id})-[r:REL {relation: $relation}]->(t:KNode {node_id: $target_id})
    WHERE r.project_id = $project_id
    RETURN r{.*} AS edge
    LIMIT 1
    """
    async with neo4j_db.session() as session:
        result = await session.run(cypher, project_id=project_id, source_id=source_id, target_id=target_id, relation=relation)
        record = await result.single()
    return _edge_from_props(record["edge"]) if record else None


async def find_edges(project_id: str) -> list[KnowledgeEdge]:
    cypher = "MATCH (:KNode)-[r:REL {project_id: $project_id}]->(:KNode) RETURN r{.*} AS edge"
    async with neo4j_db.session() as session:
        result = await session.run(cypher, project_id=project_id)
        records = [r["edge"] async for r in result]
    return [_edge_from_props(r) for r in records]


async def insert_edge(edge: KnowledgeEdge) -> KnowledgeEdge:
    cypher = """
    MATCH (s:KNode {node_id: $source_id}), (t:KNode {node_id: $target_id})
    CREATE (s)-[:REL $props]->(t)
    """
    async with neo4j_db.session() as session:
        await session.run(cypher, source_id=edge.source_id, target_id=edge.target_id, props=edge.model_dump())
    return edge


async def find_dependent_nodes(node_id: str, project_id: str, relation: Optional[KnowledgeEdgeRelation] = None) -> list[KnowledgeNode]:
    """Active nodes with a :REL edge whose target is node_id — i.e. nodes
    that depend on node_id and should be reconsidered when it changes. One
    graph traversal, replacing app.dependency_graph.find_dependents' old
    two-step "fetch matching edges, then fetch the whole project's nodes and
    filter in Python" — this is the actual payoff of the Neo4j move for this
    module."""
    cypher = """
    MATCH (s:KNode {project_id: $project_id, lifecycle: 'active'})-[r:REL]->(t:KNode {node_id: $node_id})
    WHERE $relation IS NULL OR r.relation = $relation
    OPTIONAL MATCH (s)-[:CHILD_OF]->(p:KNode)
    RETURN DISTINCT s{.*, _parent_id: p.node_id} AS node
    """
    async with neo4j_db.session() as session:
        result = await session.run(cypher, project_id=project_id, node_id=node_id, relation=relation)
        records = [r["node"] async for r in result]
    return [_node_from_props(r) for r in records]


# ---------------------------------------------------------------------------
# canonical_mapper's freeform candidate pool
# ---------------------------------------------------------------------------

_LABEL_NODE_TYPE = "Label"


async def find_label_instance_pairs(
    project_id: str, room_id: Optional[str], node_type_hint: Optional[str], *, active_only: bool = False
) -> list[tuple[KnowledgeNode, KnowledgeNode]]:
    """(label_node, instance_node) pairs for every existing freeform fact in
    this project — the candidate pool app.canonical_mapper checks before
    creating anything new. A real graph pattern match (Label -[:CHILD_OF]->
    Instance) instead of the old "fetch every node in the project, build a
    by_id dict, filter in Python" — same result, no full-project dump."""
    cypher = """
    MATCH (label:KNode {project_id: $project_id, node_type: $label_type})-[:CHILD_OF]->(instance:KNode)
    WHERE ($room_id IS NULL OR instance.room_id = $room_id OR instance.room_id IS NULL)
      AND ($node_type_hint IS NULL OR instance.node_type = $node_type_hint)
      AND (NOT $active_only OR instance.lifecycle = 'active')
    RETURN label{.*} AS label, instance{.*} AS instance
    """
    async with neo4j_db.session() as session:
        result = await session.run(
            cypher,
            project_id=project_id,
            label_type=_LABEL_NODE_TYPE,
            room_id=room_id,
            node_type_hint=node_type_hint,
            active_only=active_only,
        )
        records = [(r["label"], r["instance"]) async for r in result]
    return [(_node_from_props(label_props), _node_from_props(instance_props)) for label_props, instance_props in records]
