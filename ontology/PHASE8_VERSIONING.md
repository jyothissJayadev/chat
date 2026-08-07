# Phase 8 — Versioning: design notes

Companion to `app/versioning.py`. Per the plan: "same pattern `TraceEntry`
already uses for turn history — same shape, different collection, not a new
architectural idea." That held up in practice — this phase was small.

## What's built

`KnowledgeNodeVersion` (`app/models.py`) — append-only, one row per actual
value change. `record_version()` (`app/versioning.py`) is the only thing
that writes to it, called from the two places a `KnowledgeNode.value` is
ever set:

- `app/context_builder.py::apply_to_graph` — every structured field write
  (create or real update).
- `app/canonical_mapper.py::_create_instance` — every new freeform `Label`
  leaf.

Both call sites already had a natural "just wrote a new value" moment to
hook into — no restructuring needed beyond adding the call.

## No-ops don't get a version row

`apply_to_graph` used to bump `version` unconditionally whenever a node
already existed, even if the proposed value was identical to what was
already stored (a restated fact). Tightened this phase: `version` only
bumps, and a history row only gets written, when the value **actually
changes** (`existing.value != write.value`). A restatement that changes
nothing is a true no-op — recording it would make "what did this used to
be" queries noisy with entries that were never really different values.
Verified directly:
`tests/test_versioning.py::test_unchanged_restated_value_produces_no_new_version_row`.

Same reasoning extends to `canonical_mapper.py`'s other two save paths —
alias-list append (`map_to_canonical` reusing an existing node) and
embedding backfill — neither touches `.value`, so neither calls
`record_version`. Verified:
`test_alias_reuse_and_embedding_backfill_do_not_create_spurious_version_rows`.

## `changed_by`: only one of three values is real yet

`KnowledgeNodeVersion.changed_by` is `"user_message" | "inferred" |
"system_default"` per the plan's schema, and `ProposedWrite.changed_by`
(`context_builder.py`) carries it through. As of this phase, **every**
write actually made goes through `changed_by="user_message"` — everything
`build_context()` produces today comes from the client's own message,
structured or freeform. `"inferred"` is `app.deepinfra.infer_missing_field`'s
role in the *old* pipeline (`save_project_node`, for a field the client
never answered) — that fallback hasn't been ported to `context_builder.py`
yet, so nothing produces `"inferred"` rows today. `"system_default"` has no
producer anywhere yet either. Both are real, meaningful values the schema
is ready for — Phase 12 (Fact/Inference/Calculation split) is what actually
wires them up, not a placeholder invented here to look complete.

## `source_message_id` is still unset

Same gap noted in `PHASE6_GRAPH_SCHEMA.md` and `PHASE7_CONTEXT_BUILDER.md`:
nothing today has a message id to pass through. `record_version()` accepts
it (tested directly —
`test_record_version_captures_changed_by_and_source_message_id`) but every
real call site in `context_builder.py`/`canonical_mapper.py` leaves it
`None`. Needs an actual message-identity decision (does `app.models.Message`
gain an `id`? is it derived?) that's out of scope for the versioning
mechanism itself.

## Exit criterion, demonstrated directly

"What did the kitchen budget used to be" —
`test_changed_value_is_queryable_via_version_history`: two writes to the
same `canonical_path` with different values produce two ordered
`KnowledgeNodeVersion` rows, readable via `get_version_history(node_id)`,
independent of whatever the live `KnowledgeNode.value` currently holds.
