"""Versioning — Phase 8. record_version() is the one place that writes to
the append-only knowledge_node_versions log — mirrors TraceEntry's existing
append-only pattern for turn history (app/models.py), just a separate
collection since a node's history outlives a single chat turn. Called from
anywhere a KnowledgeNode's `value` is actually set: app/context_builder.py's
structured writes, and app/canonical_mapper.py's freeform Label creation.

See ontology/PHASE8_VERSIONING.md for what this does and doesn't cover yet."""

from typing import Optional

from app.models import ChangedBy, KnowledgeNode, KnowledgeNodeVersion, utcnow


async def record_version(
    node: KnowledgeNode, *, changed_by: ChangedBy = "user_message", source_message_id: Optional[str] = None
) -> None:
    """Appends a row for node's CURRENT value/version. Call this AFTER
    node.value/node.version already hold their new values and node itself
    is already saved — record_version only ever inserts, it never touches
    the KnowledgeNode itself."""
    await KnowledgeNodeVersion(
        node_id=node.node_id,
        version=node.version,
        value=node.value,
        changed_by=changed_by,
        source_message_id=source_message_id,
    ).insert()


async def get_version_history(node_id: str) -> list[KnowledgeNodeVersion]:
    """Full history for one node, oldest first — e.g. "what did the kitchen
    budget used to be" is `get_version_history(kitchen_budget_node_id)`,
    reading every entry's `.value` up to (not including) the current one on
    the live KnowledgeNode."""
    return await KnowledgeNodeVersion.find(KnowledgeNodeVersion.node_id == node_id).sort("+version").to_list()


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
    await node.save()
    await record_version(node, changed_by=changed_by)


async def retract_subtree(root: KnowledgeNode, *, changed_by: ChangedBy = "user_message") -> list[str]:
    """Retracts root and every node reachable via children_ids, breadth-first
    — used for room-level deletion (app.graph.delete_context_node's room-cascade
    branch). Centralized here rather than inlined in the graph node for the
    same reason app.dependency_graph.recompute_dependents centralizes its own
    traversal: a walk like this belongs in one place, not duplicated at each
    call site. Returns the node_ids actually retracted (already-retracted
    nodes reached via a shared branch are skipped, not double-counted)."""
    to_visit = [root]
    retracted: list[str] = []
    while to_visit:
        node = to_visit.pop()
        if node.lifecycle == "retracted":
            continue
        await retract_node(node, changed_by=changed_by)
        retracted.append(node.node_id)
        if node.children_ids:
            children = await KnowledgeNode.find({"node_id": {"$in": node.children_ids}}).to_list()
            to_visit.extend(children)
    return retracted
