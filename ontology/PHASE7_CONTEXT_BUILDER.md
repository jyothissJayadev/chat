# Phase 7 — Context Builder Pipeline: design notes

Companion to `app/context_builder.py`. Read this before wiring
`build_context()` into the live turn — several things below are
deliberately incomplete, by decision, not oversight.

## The Phase 15 dependency gap (why this isn't a full cutover)

The master plan's own Phase 7 text says to demote `ChatSession.partial_context`
to a small `DialogueState` (`active_room_id`, `pending_field`,
`field_attempts` only — dropping `project`, `rooms`, `field_status`). But
`PartialContext.next_field_to_ask()` — which decides what question to ask
next every turn — reads exactly the fields that demotion would remove, and
its replacement (`find_knowledge_gap`, Phase 15) depends on Phases 8, 9, and
10 existing first. Doing the demotion now would leave the live turn with no
way to decide what to ask next, for however many phases it takes to reach
Phase 15 — directly against the plan's own stated constraint that the
system stays shippable after every phase.

**Decision:** `app/context_builder.py` is built and tested standalone, same
pattern as `app/execution.py` (Phase 1) and `app/canonical_mapper.py`
(Phase 5) before it. `ChatSession.partial_context` is untouched.
`build_context()` is not called from `app/graph.py`. This is a narrower
"Phase 7" than the plan's literal text — the write path exists and is
correct, but the read-side cutover (and the `PartialContext` →
`DialogueState` demotion that depends on it) waits for Phase 15 to actually
exist, not for an arbitrary point in between.

## Conflict resolution: tier-based (confirmed with the project owner)

The plan flags `detect_conflicts`'s semantics as a product decision, not an
engineering one. Confirmed: **tier-based**, reusing `FIELD_TIERS`
(`app/models.py`) — the same critical/moderate/optional tiers that already
govern retry behavior for unanswered questions:

- **Critical** fields (`projectType`, `overallBudget`, `moreRoomsPending`,
  `roomType`, `budgetOrRequirement`) — a value that conflicts with an
  already-confirmed one is held back as a `Conflict`, not applied. Nothing
  in this phase surfaces that conflict as a question yet (no live wiring —
  see above); `BuildResult.pending_confirmations` carries it for whatever
  does that later.
- **Moderate/optional** fields (`style`, `squareFootage`, `timeline`,
  `existingFurniture`, `materials`) — a restated value auto-applies,
  overwriting in place with `version` bumped. Matches today's actual
  behavior (a restated field already just overwrites in `PartialContext`).

A "conflict" only exists when a node already has a **non-null** value that
**differs** from the newly proposed one — a first-time value, or a restated
identical value, is never a conflict regardless of tier.

## Versioning here is minimal, not Phase 8

`apply_to_graph` bumps `KnowledgeNode.version` in place and overwrites
`.value` — it does **not** write an append-only `KnowledgeNodeVersion` row.
The field existed since Phase 4 for exactly this; using it now means Phase 8
extends real behavior (an in-place bump becomes a real history entry)
instead of introducing a currently-inert field.

## Structured vs. freeform: how Gap 1/2 get enacted, not just decided

Phase 5 documented deciding that style/budget duplication becomes
structurally impossible once extraction only runs once per fact. This phase
is where that's actually true: `extract_entities` (→ `extract_fields`)
writes structured fields straight to their canonical leaf
(`Project.Rooms.<room>.Style`, etc.) — no mapper involved, no ambiguity.
`extract_relationships` (→ `extract_graph_links`) still runs concurrently
and can still propose a redundant freeform node for the same fact (the
underlying prompt wasn't touched — see below) — so **same-turn dedup** in
`build_context` drops any freeform node whose label fuzzy-matches
(`rapidfuzz.fuzz.partial_ratio >= 80`) a structured value extracted this
same turn, before it ever reaches `map_to_canonical`. This only catches
same-turn duplication; a freeform mention of something captured
structurally on an *earlier* turn isn't caught here (see canonical_mapper's
own scope note: Style/Budget are deliberately outside what it classifies
into, so its alias matching can't catch this either). Acceptable for now
since nothing is live yet; worth re-examining once it is.

**Why `_GRAPH_SYSTEM_PROMPT` wasn't changed:** it's still used by today's
*live* `update_context_graph_node` (via `deepinfra.extract_graph_links`,
the same function `extract_relationships` calls). Editing it to stop
proposing style/budget nodes would change live behavior as a side effect of
building an unwired Phase 7 capability — against the freeze rule. The
same-turn dedup filter is a self-contained alternative that touches nothing
shared.

**Room-typed freeform nodes are dropped outright, never mapped** — a
`type="room"` proposal from `extract_graph_links` is skipped before
`map_to_canonical` ever sees it. Rooms are handled exclusively through
`extract_fields` + `_resolve_room`'s fuzzy matching (`room_type_matches`,
the same matcher `PartialContext.resolve_room` already uses).

## Two things this phase does not do

**Relationship edges aren't persisted.** `extract_relationships`'
`new_edges` come back in `BuildResult.freeform_relationships` for
visibility (and are exercised by tests) but nothing writes them anywhere —
there's no `KnowledgeEdge` model yet. That's Phase 9.

**Retraction isn't honored.** `extract_graph_links` can still return
`retracted_node_ids` ("never mind the accent wall"). `build_context` logs a
warning and does nothing else — `KnowledgeNode.status` is `confirmed` /
`assumed` / `skipped` (`PartialContext`'s existing three-value semantics),
not `ContextGraph`'s `active` / `superseded` / `retracted` lifecycle.
Forcing retraction into `status` would misuse a field that means something
else. This needs an actual schema addition (a lifecycle field, or reusing
`NodeStatus` from `app/models.py`) before it can be implemented correctly —
flagged here rather than hacked around.

## Room resolution and "known" fields are sourced from `KnowledgeNode` now

`_resolve_room` reuses `room_type_matches` (fuzzy, typo-tolerant) against
existing `RoomType` leaves for the project — the same matching behavior
`PartialContext.resolve_room` has today, just queried from `KnowledgeNode`
instead. `_known_fields` builds the context shown to `extract_fields`'
prompt (project fields + the *pre-turn* active room's fields) the same way,
from `KnowledgeNode` rather than `PartialContext.active_room()`. Both are
real, tested behavior — not stubs — even though nothing calls them from a
live turn yet.
