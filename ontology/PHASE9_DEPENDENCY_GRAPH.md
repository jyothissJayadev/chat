# Phase 9 — Dependency Graph: design notes

Companion to `app/dependency_graph.py`.

## What's built

`KnowledgeEdge` — `source_id`/`target_id`/`relation`/`project_id`, using
`GraphRelation`'s vocabulary minus `part_of`/`located_in` (Gap 4,
`ontology/PHASE3_ONTOLOGY.md` — containment is already expressed by
`canonical_path`) plus a new `derives_from`, which `ContextGraph` never
needed. `add_edge` is idempotent on the `(source, target, relation)` triple,
matching `update_context_graph_node`'s existing dedup convention.

`find_dependents(node_id, project_id, relation=None)` — nodes with an edge
targeting `node_id`. `recompute_dependents(changed_node, project_id,
recompute_fn)` — walks `derives_from` dependents specifically, calls a
caller-supplied `recompute_fn` for each, and **writes the result itself**
(value, version bump, `changed_by="system_default"` history row via Phase 8
— corrected in Phase 12: a deterministic recalculation is `system_default`,
not `inferred`, which means an LLM guess specifically) so a
caller only ever supplies the calculation, never touches persistence.

## No business rule lives here — deliberately

The plan's own worked example ("cabinet budget derives_from kitchen budget")
implies a real calculation (e.g. "cabinet budget is N% of the room
budget"). No such rule exists anywhere in this codebase — there's no
quotation-generation logic at all yet (`Project.Quotation` in
`ontology/v1.yaml` is explicitly reserved, unpopulated). Inventing a
percentage rule inside `app/dependency_graph.py` would be fabricating
product logic this repo doesn't have, dressed up as infrastructure. Instead,
`recompute_fn` is a plain callback the *caller* supplies — the mechanism is
real and tested (`tests/test_dependency_graph.py`, using a simple "15% of
parent" rule defined in the test itself, clearly not app logic), but no
actual derivation rule ships until a real one exists to write.

`recompute_fn` returning `None`, or a value identical to the dependent's
current one, is a no-op — no version bump, no history row (same "no
unchanged writes" discipline Phase 8 already established for `apply_to_graph`).

## Sync vs. async: still an open wiring decision

The plan recommends async recompute via the existing `context_updated`
event, since the turn already has a slot for it. That's a **live-wiring**
decision — nothing calls `recompute_dependents` from a real turn yet (same
standing as every module since Phase 1), so there's nothing to hook an
event into. `recompute_dependents` itself is a plain `await`-able function;
whoever wires it into a live turn later decides whether that happens inline
or is queued.
