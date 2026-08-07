# Phase 6 — Graph Schema: gap-check

Per the implementation plan, Phase 6 is "largely already done by Phase 4's
`KnowledgeNode` model" — this doc confirms that and records the two fields
the plan's exit criteria explicitly calls for adding.

## Field-by-field against the plan's "Rewrite Context Storage" list

| Plan's field | `KnowledgeNode` field | Status |
|---|---|---|
| `id` | `node_id: str` | Present since Phase 4 — a business key, not Beanie's own `id`; see the class docstring for why. |
| `canonical_path` | `canonical_path: str` | Present since Phase 4. |
| `parent` | `parent_id: Optional[str]` | Present since Phase 4. |
| `children` | `children_ids: list[str]` | Present since Phase 4. |
| `aliases` | `aliases: list[str]` | Present since Phase 4. |
| `confidence` | `confidence: float` | Present since Phase 4. |
| `source` | `source_message_id: Optional[str]` | Present since Phase 4 (not yet set by anything — `map_to_canonical()`'s signature has no message-id parameter; Phase 7's context builder, which has the actual message context, is the natural place to populate it). |
| `created` | `created_at: datetime` | Present since Phase 4. |
| `updated` | `updated_at: datetime` | Present since Phase 4. |
| `status` | `status: FieldStatus` | Present since Phase 4 — reuses `PartialContext`'s existing three-value enum rather than inventing a parallel one. |
| `version` | `version: int` | Present since Phase 4 (still `1` for every node — Phase 8 is what actually starts incrementing it via `KnowledgeNodeVersion`). |
| `tenant` | `tenant_id: Optional[str]` | **Added this phase.** Unused — this system has exactly one tenant — but indexed now per the plan's own stated rationale: retrofitting tenant scoping onto an already-growing collection later is materially harder than shipping an unused, indexed column today. |
| `project` | `project_id: str` | Present since Phase 4. |
| `room` | `room_id: Optional[str]` | Present since Phase 4. |
| `embedding` | `embedding: Optional[list[float]]` | **Added this phase.** Populated by `app/canonical_mapper.py` (Phase 5, retrofitted this phase) — see below. |

No gaps: every field in the plan's list has a `KnowledgeNode` counterpart.
Two extra fields exist beyond the plan's list — `node_type` (which
ontology/v1.yaml node this is; needed to do anything useful with
`canonical_path` at all) and `value` (the leaf's actual data) — both
load-bearing since Phase 4, not scope creep introduced now.

## `embedding`: populated, not just declared

The plan's exit criteria says "add `embedding: list[float] | None` field
(populated by the Phase 5 mapper)" — added as a bare field without wiring it
up would leave Phase 5's own docstring (`ontology/PHASE5_CANONICAL_MAPPER.md`,
"Why Python-side similarity") stating a deliberate decision *not* to persist
embeddings, now contradicted by the schema. Rather than ship a dead field,
`app/canonical_mapper.py` was revisited this phase:

- A newly created `Label` node's embedding (the query vector already
  computed to classify it) is persisted directly — no extra embedding call
  spent just to save it.
- An existing candidate's persisted embedding is reused on later
  `map_to_canonical()` calls instead of being re-embedded every time; only a
  candidate with no persisted embedding yet (e.g. a `Label` written by
  `scripts/migrate_partial_context_to_tree.py`, which doesn't set this
  field) gets backfilled, lazily, the first time it's considered.
- The five ontology type descriptions (`ontology/v1.yaml`) are static for
  the process lifetime — embedded once and cached module-wide rather than
  re-embedded on every type-inference call.

This turns "populated by the Phase 5 mapper" from a schema note into
something the mapper's own tests verify
(`tests/test_canonical_mapper.py::test_existing_candidate_embedding_is_persisted_and_reused_not_recomputed`).

## What's still not done, deliberately

- `KnowledgeNodeVersion` (append-only value history) is Phase 8, not this
  phase — `version` stays `1` everywhere for now.
- `source_message_id` stays unset — nothing today has a message id to pass
  it. Phase 7's context builder is where a live turn's message id becomes
  available to thread through.
- `tenant_id` stays unset everywhere it's written (`scripts/migrate_partial_context_to_tree.py`,
  `app/canonical_mapper.py`) — there's exactly one tenant today, so setting
  it to anything would be inventing a value with no meaning yet.
