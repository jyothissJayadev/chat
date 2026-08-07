# Phase 14 — Graph Reasoning: design notes

Companion to `app/graph_reasoning.py`. Three precomputed questions, per the
plan, each answerable purely from `KnowledgeNode`/`KnowledgeEdge` — zero
`ChatSession.messages` reads.

## `rooms_exceeding_budget` needed a real design gap resolved

`Project.Quotation.RoomLineItems` (`ontology/v1.yaml`) was added in Phase 3
as a reserved placeholder — a `children: [RoomLineItems]` entry under
`Quotation`, never fully designed since nothing populated it. Writing this
function surfaced the gap: `RoomLineItems` isn't path-nested under
`Rooms.<instance>` (it's a sibling subtree, `Project.Quotation.RoomLineItems.<item>`),
so there was no defined way to know *which room* a line item belongs to
from its `canonical_path` alone.

**Resolved:** a `RoomLineItems` instance is tagged to its room via
`KnowledgeNode.room_id` — the same metadata field every `Rooms`-nested node
already carries, just used here for a node that isn't path-nested under the
room it concerns. This is additive (no `ontology/v1.yaml` path renamed) and
consistent with `room_id` already being cross-cutting metadata elsewhere,
not solely derived from path structure.

Nothing writes `RoomLineItems` yet — no quotation-generation code exists.
Tested with directly-seeded data (`tests/test_graph_reasoning.py`'s
`_room_line_item` helper), same posture as `app/dependency_graph.py`'s own
worked-example tests: the *query* is real, the *data it reads* doesn't
exist in production yet.

## Budget comparison is best-effort, not a currency parser

Both `Budget` and `Amount` values are free text (`"$8k"`, `"$5,000"`, a
plain number). `_parse_amount` extracts the first numeric run and treats an
immediately-following `"k"` as ×1000 — enough for the test fixtures and
plausible real input, not a robust currency parser (none exists anywhere in
this codebase). A room whose budget or every line item fails to parse is
**skipped**, not guessed at — matches the rest of this codebase's "don't
silently invent a number" posture (`_is_unset`, extraction salvage, etc.).

## The other two questions were mechanical

`rooms_with_material` — fuzzy match (`rapidfuzz`, threshold 85 — stricter
than `context_builder`'s same-turn dedup threshold of 80, since this is a
direct answer to "which rooms have X," not a heuristic guard) over every
`Material` leaf in the project, deduped to one `RoomRef` per room.

`items_depending_on` — a thin, purpose-named wrapper over
`app.dependency_graph.find_dependents` (Phase 9). "What depends on this
material" is exactly "what has a `derives_from` edge targeting it" — no new
logic needed, just a name matching the question being asked.
