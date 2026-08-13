"""Dependency Graph — Phase 9. A KnowledgeEdge records a non-hierarchical
relationship between two KnowledgeNodes; containment ("this is in the
kitchen") is already expressed by canonical_path itself (Gap 4,
ontology/PHASE3_ONTOLOGY.md), so part_of/located_in are deliberately not
part of this vocabulary.

This module is the GENERIC recompute mechanism — find what depends on a
changed node, re-run a caller-supplied rule for each, record the result
through the normal versioning path (Phase 8). It contains no actual
business rule (e.g. "cabinet budget is N% of the room budget"). No such
rule exists anywhere in this codebase yet, and inventing one here would be
fabricating quotation logic that belongs to a future phase, not to
dependency-graph plumbing — see ontology/PHASE9_DEPENDENCY_GRAPH.md for the
worked example used to test the mechanism instead.

Not wired into the live turn — same standing as every module since
app/execution.py (Phase 1)."""

from typing import Awaitable, Callable, Optional

from app import graph_store
from app.models import KnowledgeEdge, KnowledgeEdgeRelation, KnowledgeNode
from app.versioning import record_version


async def add_edge(source_id: str, target_id: str, relation: KnowledgeEdgeRelation, project_id: str) -> KnowledgeEdge:
    """Idempotent — re-adding an identical (source, target, relation) tuple
    returns the existing edge rather than creating a duplicate, same
    dedup convention update_context_graph_node already uses for
    (source, target, relation) triples."""
    existing = await graph_store.find_edge(project_id, source_id, target_id, relation)
    if existing is not None:
        return existing
    edge = KnowledgeEdge(source_id=source_id, target_id=target_id, relation=relation, project_id=project_id)
    return await graph_store.insert_edge(edge)


async def find_dependents(node_id: str, project_id: str, relation: Optional[KnowledgeEdgeRelation] = None) -> list[KnowledgeNode]:
    """Nodes with an edge whose target_id == node_id — i.e. nodes that
    depend on node_id and should be reconsidered when node_id's value
    changes. Defaults to every relation; pass relation="derives_from" to
    scope to genuine dependency edges specifically (as recompute_dependents
    does) rather than e.g. "modifies" or "requires" edges, which describe a
    relationship without implying a value should be recalculated.

    One graph traversal (graph_store.find_dependent_nodes) — the old
    two-step "fetch matching edges, then fetch every active node in the
    project and filter by source_id in Python" collapses into a single
    Cypher MATCH, which is the actual payoff of the Neo4j move for this
    module."""
    return await graph_store.find_dependent_nodes(node_id, project_id, relation=relation)


RecomputeFn = Callable[[KnowledgeNode, KnowledgeNode], Awaitable[Optional[object]]]


async def recompute_dependents(changed_node: KnowledgeNode, project_id: str, recompute_fn: RecomputeFn) -> list[KnowledgeNode]:
    """Walks every node with a "derives_from" edge targeting changed_node and
    calls recompute_fn(dependent, changed_node) for each. recompute_fn
    decides the new value (returning None leaves the dependent untouched);
    this function does the writing — value, version bump, and a
    changed_by="system_default" history row (Phase 8, refined by Phase 12 —
    a deterministic recalculation is system_default, not inferred; inferred
    means an LLM guess, see app/inference.py) — so a caller only supplies
    the calculation, never touches persistence directly. Returns the
    dependents that were actually updated (skips any recompute_fn returned
    None for, or whose result was unchanged from the current value).

    Whether this runs synchronously inline with the turn that changed
    changed_node, or is queued to run after (the plan's own recommendation,
    via the existing context_updated event), is a live-wiring decision with
    nothing to wire into yet — see ontology/PHASE9_DEPENDENCY_GRAPH.md."""
    dependents = await find_dependents(changed_node.node_id, project_id, relation="derives_from")
    updated: list[KnowledgeNode] = []
    for dependent in dependents:
        new_value = await recompute_fn(dependent, changed_node)
        if new_value is not None and new_value != dependent.value:
            dependent.value = new_value
            dependent.changed_by = "system_default"
            dependent.version += 1
            await graph_store.save_node(dependent)
            await record_version(dependent, changed_by="system_default")
            updated.append(dependent)
    return updated
