# Phase 11 — Multi-Task Executor hardening: design notes

Companion to `app/execution.py`. Per the plan, this phase is "largely
delivered already by Phase 1's `execute()` and Phase 2's `TaskSpec[]`" — the
hardening pass, not new architecture. Two real gaps found and fixed.

## Partial-failure isolation (the actual gap)

`execute()` dispatched its batch via a bare `asyncio.gather(...)` — no
`return_exceptions=True`. In practice this rarely mattered, because the
individual node functions it dispatches to (`extract_fields_node` via
`_run_update_context`, etc.) already swallow their own extraction failures
internally (existing resilience, covered by `tests/test_graph.py`). But
*something* dispatched here failing for an unrelated reason — a bug, a
database hiccup, anything not already caught one level down — would have
taken the whole batch's results with it, including a sibling task that
succeeded. Fixed: `return_exceptions=True`, with each failure logged and
recorded as a `TraceEntry` (`"failed: <type>: <message>"`, matching the
shape `extract_fields_node` already uses for its own internal failures) so
a failure is visible in the trace, not silent. Tested by patching
`app.graph._run_update_context` directly (not the LLM call underneath it,
which the existing resilience already absorbs) to force an actual
execute()-level failure.

## Response merging

`merge_task_results(parts: list[str]) -> str` — new, in `app/execution.py`,
backed by a new `deepinfra.merge_response` (same fast/low-stakes model
`generate_wrapup_message` already uses, not the full answer model). A
single non-empty part returns unchanged with **no model call** — nothing to
merge. Two or more real parts get blended into one natural-language message
by the model rather than concatenated or left as separate strings.

**Not called from `app/chat.py`.** Today's live turn already sends
`update_summary`/`wrapup`/`ask_question` as separate SSE events by design
(see that module's event-shape docstring) — merging them into one message
is a real behavior change to what the client receives, not a pure
internal refactor like everything else built standalone-but-unwired so far.
Building the capability and deciding to switch the live turn's message
shape over to it are different things; only the former is this phase's job.
