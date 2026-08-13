# `pending_gap` — Full Lifecycle & Connection Map

> Updated for the pending_gap-centralization refactor (see the plan at
> `C:\Users\jyoth\.claude\plans\dapper-popping-abelson.md`). This revision
> replaces the original analysis, which documented the pre-refactor design
> (7+ read/write sites, `active_room_id` fallback, rephrase-then-infer
> decline handling). That design is gone; this document describes what
> replaced it.

This document traces where `pending_gap` appears in the codebase today, what
it means, and how it connects to the two new fields that took over the jobs
it used to do itself: `skipped_rooms` and `current_field`.

## 1. What it is

`pending_gap` is the **serialized form of a `KnowledgeGap`** — the record of
"which project field is currently missing." It is not a project fact (it
never becomes a `KnowledgeNode`); it's session-level dialogue mechanics.

Defined in `app/question_engine.py`:

```python
class KnowledgeGap(BaseModel):
    canonical_path: str      # e.g. "Project.Rooms.r1.Style" or "Project.BasicInformation.ProjectType"
    field_label: str         # human-readable label, e.g. "style"
    node_type: str           # e.g. "Style", "ProjectType", "Total"
    room_id: Optional[str]   # set only for room-scoped fields
```

No `tier` field anymore — tiering only ever fed the old rephrase-then-infer
retry budget (`RETRY_LIMITS`), which is gone (see §4).

## 2. The single-writer / single-reader contract

Unlike the old design (four different nodes independently computing and
stashing a gap), `pending_gap` now has exactly two writers in
`app/graph.py`, both in the same module:

- **`build_context_node`** — on a successful commit (no held-back conflict),
  it recomputes `find_knowledge_gap(project_id, skipped_rooms)` right after
  the write lands and stores the result. This keeps the cache in sync with
  facts the instant they change, rather than waiting for the next question
  to be asked.
- **`generate_question_node`** — the single consolidated question-generation
  node (see §5). It reads the cached value; if there's nothing cached yet
  (bootstrap: the very first gap of a project) or the cached gap's room was
  *just* added to `skipped_rooms` this same turn (a decline would otherwise
  make it re-ask about the room the user just declined), it recomputes via
  `find_knowledge_gap` instead of trusting the cache. Either way, it writes
  back whatever gap it ends up asking about.

No other node reads or writes `pending_gap`. `classify_intent_node` —
which used to read it for classifier context — now reads `current_field`
instead (see §4).

```
KnowledgeGap (question_engine.find_knowledge_gap)
        │ .model_dump()
        ▼
GraphState["pending_gap"]  — written by build_context_node + generate_question_node only
        │ round-tripped by app/chat.py::run_chat_turn
        ▼
ChatSession.pending_gap  — persisted in Mongo
        │ exposed read-only
        ▼
GET /sessions/{id}/state  (app/routes.py)
```

Known limitation, accepted as part of the plan: `confirm_conflict_node` and
`delete_context_node` can also change facts (apply a confirmed edit, retract
a node) but don't refresh `pending_gap`. In practice this rarely surfaces a
wrong question — those paths only ever *widen* the gap set or overwrite an
already-set value, essentially never closing the exact gap currently cached.

## 3. `find_knowledge_gap`'s new walk order

`app/question_engine.py::find_knowledge_gap(project_id, skipped_rooms=())`:

1. `Project.BasicInformation.ProjectType` — blocking.
2. Room existence (at least one room must exist) — blocking.
3. Room fields, **breadth-first, two passes**:
   ```
   ordered_rooms = rooms sorted by their container node's created_at
   for candidate_rooms in (non_skipped_rooms, skipped_rooms_subset):
       for node_type, field_name in _ROOM_FIELD_ORDER:   # RoomType, Budget, Style, SquareFootage, ExistingFurniture
           for room_id in candidate_rooms:
               if open(...): return that gap
   ```
   Every non-skipped room is asked about field *N* before any room is asked
   about field *N+1*. Skipped rooms are only considered in the second pass,
   once every non-skipped room's fields are all filled.
4. `Project.Timeline.Value`.
5. `Project.Budget.Total` — moved here from position 2.
6. `None` (complete).

A room is never explicitly "unskipped." Once every non-skipped room is
exhausted, skipped rooms simply become the only candidates left in the
second pass — a standing low-priority tier, not a temporary flag.

## 4. `skipped_rooms` and `current_field` — what replaced retry tracking

The old rephrase-then-infer mechanic (`decline_field_node`, `RETRY_LIMITS`,
`field_attempts`, an `inference.infer_field` call on budget exhaustion) is
gone entirely. Decline is now a **non-terminal state mutation**, not a
routing branch:

- **`current_field: Optional[dict]`** (`{"canonical_path", "room_id"}`) —
  the field the *last* question was about. Written only by
  `generate_question_node`. Read only by `classify_intent_node`, for two
  things: feeding `classify_operations`' "currently pending field" prompt
  context (what `pending_gap` used to be read for — see the original
  analysis's §7), and decline-detection.
- **`skipped_rooms: list[str]`** — rooms currently deprioritized. Mutated
  only by `classify_intent_node`: if the incoming message matches
  `_is_decline()` and `current_field["room_id"]` is set, that room is added.
  This does **not** short-circuit the turn — `_route_intent` no longer has a
  `"decline_field"` branch at all. The turn's normally-classified operation
  (`DIRECT_ANSWER`/`CONTEXT_UPDATE`/etc.) still runs exactly as it would
  have; `generate_question_node` picks up the updated `skipped_rooms`
  afterward and naturally asks about something else.
- For the three non-room-scoped blocking gaps (ProjectType, room existence,
  Budget.Total), a decline is simply a no-op — there's no room to skip, so
  the same question just gets re-asked next turn, unchanged.

Both fields round-trip through `ChatSession` the same way `pending_gap`
always has (`app/chat.py` seeds `GraphState` from `session.*` at turn start,
syncs back at turn end).

## 5. The single consolidated question-generation node

The old separate `generate_question_node` and `analyze_context_node` (a
near-duplicate pair — one reached only from `validate_completeness`'s
"incomplete" branch, the other a catch-all running after almost every other
branch) are merged into one `generate_question_node`. Every edge that used
to target `"analyze_context"` now targets `"generate_question"` — there is
exactly one node in the graph responsible for turning a `KnowledgeGap` into
an actual question.

Its guard (`question_generated`) is unchanged: if some other node already
produced this turn's trailing ask (a conflict confirmation, an
operation-clarification batch, a delete's ambiguous-match question), this
node no-ops.

## 6. Relationship to the other two "pending" fields

| Field | Set by | Represents |
|---|---|---|
| `pending_confirmation` | `build_context_node`, `delete_context_node` | A critical-tier value change/deletion awaiting yes/no |
| `pending_operation_questions` | `classify_intent_node` | Write ops the classifier couldn't ground to a room |
| `pending_gap` | `build_context_node`, `generate_question_node` | The current missing field |

`app/chat.py::run_chat_turn` still persists exactly one of these three per
turn (mutual exclusion, priority order: confirmation > operation-questions >
plain question), nulling the other two — unchanged from before.

## 7. `active_room_id` — removed system-wide

The `active_room_id` session field is gone entirely (`GraphState`,
`ChatSession`, `app/chat.py`, `app/routes.py`, `app/execution.py`,
`app/context_builder.py`, `app/graph.py`). Room grounding for a write now
comes **only** from `task.connection`/`task.room_hint` (already always
required to be resolved before a write task reaches `build_context_node`/
`delete_context_node` — see the classifier-connection-always-block
behavior). `app/context_builder.resolve_context`'s existing "create an
unlabeled room" fallback (unchanged) absorbs the case where a room-scoped
fact has no room named at all — so no data is ever silently dropped, but a
follow-up message with no explicit room mention now creates a **new**
unlabeled room rather than continuing whichever room the previous turn
touched. No replacement for that continuity was designed, per an explicit
scope decision.

`app/static/viewer.html`'s debug panel now highlights a room as "(asking)"
when it matches `current_field.room_id`, and marks a room "(skipped)" when
its `room_id` is in `skipped_rooms` — the closest equivalents to the old
"(active)" label.
