"""Versioning — Phase 8. record_version() is the one place that writes to
the append-only KNodeVersion log — mirrors TraceEntry's existing append-only
pattern for turn history (app/models.py), just a separate set of graph
nodes since a node's history outlives a single chat turn. Called from
anywhere a KnowledgeNode's `value` is actually set: app/context_builder.py's
structured writes, and app/canonical_mapper.py's freeform Label creation.

See ontology/PHASE8_VERSIONING.md for what this does and doesn't cover yet."""

from typing import Optional

from app import graph_store
from app.models import ChangedBy, KnowledgeNode, KnowledgeNodeVersion, utcnow


async def record_version(
    node: KnowledgeNode, *, changed_by: ChangedBy = "user_message", source_message_id: Optional[str] = None
) -> None:
    """Appends a row for node's CURRENT value/version. Call this AFTER
    node.value/node.version already hold their new values and node itself
    is already saved — record_version only ever inserts, it never touches
    the KnowledgeNode itself."""
    await graph_store.insert_version(
        KnowledgeNodeVersion(
            node_id=node.node_id,
            version=node.version,
            value=node.value,
            changed_by=changed_by,
            source_message_id=source_message_id,
        )
    )


async def get_version_history(node_id: str) -> list[KnowledgeNodeVersion]:
    """Full history for one node, oldest first — e.g. "what did the kitchen
    budget used to be" is `get_version_history(kitchen_budget_node_id)`,
    reading every entry's `.value` up to (not including) the current one on
    the live KnowledgeNode."""
    return await graph_store.find_versions(node_id)


async def retract_node(node: KnowledgeNode, *, changed_by: ChangedBy = "user_message") -> None:
    """Flips a node's lifecycle to "retracted" and appends a version row for
    it — the deletion counterpart to app.context_builder.apply_to_graph's
    update branch. Deliberately does not touch `value`: the last real value
    stays on the node (and in its version history) for audit purposes, only
    `lifecycle` changes, which every read path now filters on (see
    app.question_engine.find_knowledge_gap, app.context_builder.known_fields,
    etc.). A no-op if the node is already retracted — callers (e.g.
    retract_subtree below) don't need to check first."""
    if node.lifecycle == "retracted":
        return
    node.lifecycle = "retracted"
    node.version += 1
    node.updated_at = utcnow()
    await graph_store.save_node(node)
    await record_version(node, changed_by=changed_by)


async def retract_subtree(root: KnowledgeNode, *, changed_by: ChangedBy = "user_message") -> list[str]:
    """Retracts root and every node reachable via :CHILD_OF, breadth-first —
    used for room-level deletion (app.graph.delete_context_node's
    room-cascade branch). The descendant walk itself is one graph query
    (graph_store.descendant_ids) rather than re-fetching each generation by
    id; retraction still happens node-by-node through retract_node so every
    node gets its own version row. Returns the node_ids actually retracted
    (already-retracted nodes are skipped, not double-counted)."""
    ids = await graph_store.descendant_ids(root.node_id)
    retracted: list[str] = []
    for node_id in ids:
        node = root if node_id == root.node_id else await graph_store.find_by_node_id(node_id)
        if node is None or node.lifecycle == "retracted":
            continue
        await retract_node(node, changed_by=changed_by)
        retracted.append(node_id)
    return retracted
