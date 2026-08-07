# Phase 10 — Scoped Retrieval Engine: design notes

Companion to `app/retrieval.py`. Folds in the plan's separate "Phase 13"
line item — "never send the whole graph, send relevant paths only" is
exactly this module, not additional work (the plan says as much itself).

## What's built

`resolve_query_to_path` — fuzzy string match (`rapidfuzz`, not embeddings:
matching a query against a small set of already-known room names has no
real classification ambiguity for `app/canonical_mapper.py`-style inference
to resolve) against existing room names. A named room resolves to
`Project.Rooms.<room_id>`; no room detected falls back to `Project` (the
whole tree, but still depth-limited below).

`load_subtree(project_id, root_path, depth=2)` — every node at or under
`root_path` within `depth` canonical_path segments, via simple path-prefix
and segment-count filtering over the project's nodes (fetched once — same
Python-side, not `$vectorSearch`-style, scale justification
`app/canonical_mapper.py` already documents: one project's node count is
small and bounded).

`retrieve_scoped` composes the two — this is what replaces `app/graph.py`'s
`retrieve_context_node`.

## Exit criterion, demonstrated with a proxy

"Token count sent to generate_answer drops measurably... without a drop in
answer quality on a held-out eval set." No live model call exists to
measure real tokens or run an eval set against (nothing's wired up — same
standing as every phase since Phase 1). Approximated instead:

- **Token count** → character count of the serialized node set (the same
  proxy `app/graph.py::retrieve_context_node` already uses today — it joins
  a summary string, never calls a tokenizer either). Tested directly: a
  three-room project scoped to one room sends well under half the
  characters of the unscoped whole-project payload.
- **Answer quality** → "the actual fact being asked about survives
  scoping," the closest thing to quality that's checkable without a live
  model: `test_retrieve_scoped_excludes_other_rooms_but_keeps_the_answer`
  confirms the kitchen's own budget node is present in a kitchen-scoped
  query's result, not just that other rooms were excluded.

Real eval-set validation is a live-wiring concern, same gap already
documented for the token-count measurement itself.
