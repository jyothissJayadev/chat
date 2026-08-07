# Phase 12 — Fact / Inference / Calculation split: design notes

Companion to `app/facts.py`, `app/inference.py`, `app/calculation.py`.

## `changed_by` moved onto `KnowledgeNode` itself

Phase 8 only recorded provenance on `KnowledgeNodeVersion` (the history
log). Filtering "which of this project's values are facts" would have
meant walking every node's history to check its latest entry — an N+1
query pattern. Denormalized `changed_by` onto `KnowledgeNode` too, the same
"fast field on the live node, full history in the append-only log" split
`value` itself already uses. Kept in sync by every write path:
`app.context_builder.apply_to_graph`, `app.dependency_graph.recompute_dependents`,
and the two new write paths below.

**Found and fixed while doing this:** `app.dependency_graph.recompute_dependents`
(Phase 9) was recording its cascaded recalculations as `changed_by="inferred"`.
That's wrong under this phase's own taxonomy — a deterministic recompute
(pure math, no LLM) is `"system_default"`; `"inferred"` specifically means
an LLM's best-guess fallback. Corrected in both `app/dependency_graph.py`
and its tests.

## Three namespaces, each earning its place

- **`app/facts.py`** — `list_facts()` only. The *write* side needed no new
  code: `apply_to_graph`/`_create_instance` already default to
  `changed_by="user_message"`, correctly, since that's genuinely where
  every value they write comes from. Excludes `value is None` nodes —
  `ensure_path`'s structural containers (`Project`, `Project.Rooms`, ...)
  also default to `"user_message"` but aren't facts in any meaningful sense,
  just scaffolding.
- **`app/inference.py`** — `infer_field()` finally ports
  `infer_missing_field`'s *role* into this pipeline. `save_project_node`
  (the old system) already does this when intake completes; Phase 7 didn't
  port it (documented gap). Not called from `build_context()` yet — nothing
  there decides "the client is done answering, fill in what's left"; that's
  completeness logic, Phase 15's job. `list_inferred()` for the read side.
- **`app/calculation.py`** — `calculate_and_record()`, `changed_by="system_default"`.
  No calculation rule lives here, same reasoning as `app/dependency_graph.py`
  (Phase 9): no quotation/pricing logic exists in this codebase yet to
  encode. `list_calculated()` for the read side.

## Exit criterion, demonstrated generically

"The quotation output can filter/label by this field" — no quotation
feature exists to test against (`Project.Quotation` is reserved,
unpopulated). Demonstrated instead with the underlying mechanism a
quotation feature would actually use:
`test_all_three_provenances_partition_a_projects_nodes_without_overlap`
writes one fact, one inferred value, and one calculated value into the same
project and confirms `list_facts`/`list_inferred`/`list_calculated`
partition them correctly — no node counted twice, none missing.
