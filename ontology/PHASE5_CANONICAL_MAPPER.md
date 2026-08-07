# Phase 5 — Canonical Mapping Engine: design notes

Companion to `app/canonical_mapper.py`. Not wired into the live turn —
see `ARCHITECTURE_BASELINE.md`'s freeze rule; that's Phase 7's job.

## Scope

Only the five freeform, instantiable "fact bucket" ontology types go through
`map_to_canonical()`: `Materials`, `Furniture`, `Attributes`, `Constraints`,
`ClientPreferences`. Structured slot fields (`ProjectType`, `Budget`,
`Style`, `RoomType`, `SquareFootage`, `ExistingFurniture`, `Timeline`) are
populated by `app.deepinfra.extract_fields`'s already-deterministic schema —
there's no classification ambiguity for a mapper to resolve there, so
routing them through this module would be solving a problem that doesn't
exist for them. This is also how Gaps 1 and 2 (`PHASE3_ONTOLOGY.md`) get
resolved: once Phase 7 retires today's *second*, independent
`extract_graph_links` call, a style or budget statement is only ever
extracted once, by `extract_fields`, straight into its structured slot —
there's no freeform "preference"/"budget" node for the mapper to even see,
so the duplication is structurally impossible rather than merely avoided.

## Matching pipeline

For a raw entity mention (`raw_entity`), an optional caller-supplied
`node_type_hint`, `project_id`, and optional `room_id`:

1. **Exact alias lookup** — case/whitespace-normalized string match against
   every existing `Label` node's `value` and `aliases`, scoped to the
   project (and room, and type hint, if given). Free — no model call.
2. **Embedding similarity against existing labels** — if step 1 finds
   nothing, embed `raw_entity` plus every remaining candidate's `value` in
   one batched call (`deepinfra.embed`, the same model already wired for
   catalog search — see `app/rag.py`), score by cosine similarity, and reuse
   the best match if it clears `_MATCH_THRESHOLD`. This is what resolves
   "TV Cabinet" → the same node as an existing "TV Unit": different words,
   high embedding similarity, no exact string match. Reusing a node appends
   the new mention as an alias, so a *third* mention ("entertainment unit")
   can resolve via the cheaper exact-match step next time.
3. **Type inference** — if nothing existing matched well enough, and no
   `node_type_hint` was given, embed `raw_entity` against each of the five
   node types' `description` in `ontology/v1.yaml` (written specifically to
   be distinguishable — see that file) and take the best-scoring type. This
   is Gap 3's resolution: a material-shaped mention like "quartz countertop"
   now scores higher against `Materials`' description than `Furniture`'s,
   instead of defaulting to whatever the extraction model's catch-all type
   happened to be.
4. **Unmapped fallback** — if the best type score is still below threshold,
   *or* the winning type needs a room (`Materials`/`Furniture`/`Attributes`)
   and none was given, the mention is created under `Unmapped` instead
   (room-scoped if a room *is* known, project-level otherwise — see
   `ontology/v1.yaml`'s comment on `Unmapped`) and flagged for review. This
   never invents a path outside `ontology/v1.yaml` — `Unmapped` is itself a
   reviewed node type, not an ad hoc escape hatch.

Every outcome is reported via `CanonicalMatch.matched_via`
(`alias_exact` / `alias_embedding` / `type_hint` / `type_embedding` /
`unmapped`) and `.confidence`, so a caller (or a future review dashboard)
can distinguish "confidently reused," "confidently created," and "flagged"
without re-deriving it.

## Why one threshold, and why 0.75

`_MATCH_THRESHOLD` gates two different questions — "is this the same
existing entity" (step 2) and "is this type classification trustworthy"
(step 3) — with a single constant for now, rather than two independently
tuned ones. `0.75` is a documented starting point, **not measured against
real embeddings from this system** — there's no production traffic yet to
tune against (this is a fresh repo; see `ARCHITECTURE_BASELINE.md`). Splitting
the threshold, or tuning either value, is a natural follow-up once the
"created new node, low confidence" rate (`flagged_for_review=True`) is
observable against real usage — tracking that rate in Langfuse, per the
original plan, is Phase 7 wiring, not something to fake here.

## Why Python-side similarity, not Atlas `$vectorSearch`

`app/rag.py`'s catalog search uses MongoDB Atlas `$vectorSearch` because the
catalog is an open-ended, potentially large collection. A single project's
freeform-fact count (`Materials`/`Furniture`/`Attributes`/`Constraints`/
`ClientPreferences` instances) is bounded by nature — a few dozen at most
for one interior design project — so fetching the whole project's
`KnowledgeNode` rows in one query and comparing embeddings in plain Python
(`_cosine_similarity`, no new dependency) is simpler, equally correct, and
works identically in tests, local dev, and production, unlike
`$vectorSearch`, which only works against a real Atlas cluster (see the
`try`/`except OperationFailure` fallback already in
`app/database.py::_ensure_indexes` for the catalog index). No embeddings are
persisted on `KnowledgeNode` for this reason — they're computed fresh per
call over a small, already-fetched candidate set. Persisting embeddings for
reuse is Phase 6's job if it turns out to matter at real scale, not pulled
forward here.

## Node creation

A newly created instance is not a bare leaf: `map_to_canonical()` writes
both the instance node (e.g. `Project.Rooms.<room>.Furniture.<slug>`) and a
`Label` child holding `raw_entity` as its `value` — the same shape
`scripts/migrate_partial_context_to_tree.py` (Phase 4) now also produces for
`Materials` instances, so every freeform-fact instance in the tree,
regardless of which phase created it, is uniformly findable by the same
alias/embedding search this module runs. Any missing ancestor container
(`Project.Rooms`, the room instance itself, `Project.Requirements`, etc.) is
created on demand (`_ensure_path`) rather than assumed to already exist —
necessary because Phase 4's migration never touches the freeform layer at
all (see Gap 5), so a project migrated by Phase 4 alone has no `Furniture`/
`Attributes`/etc. containers yet.
