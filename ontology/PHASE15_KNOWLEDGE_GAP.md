# Phase 15 — Knowledge-Gap Question Generation: design notes

Companion to `app/question_engine.py`. Last phase in the plan's numbered
sequence.

## Scope: project-level completeness, not task-scoped

The plan's `find_knowledge_gap(project_id, current_task: TaskSpec)` walks
exactly what a *specific* task (its example: `QUOTE_GENERATION`) needs.
That's not buildable today — `QUOTE_GENERATION` (along with `SAVE_CONTEXT`,
`DELETE_CONTEXT`, `MEMORY`) was deliberately never added to
`app.tasks.TaskType` (see that module's own docstring — nothing implements
them, adding the enum values would have been dead schema). With no
per-task dependency subtree to walk, `find_knowledge_gap` here answers a
related, real, useful question instead: **"what's still missing for this
project overall"** — a direct, faithful port of
`PartialContext.next_field_to_ask()`'s actual job, just walking
`KnowledgeNode` state against the ontology instead of a hardcoded
`PROJECT_FIELDS`/`ROOM_FIELDS` list.

## The tiering decision was already forced

The plan flags a product decision: does retry-tier depend on the node's
position in the ontology, or on how many pending tasks currently need it?
The second option has nothing to be based on — there's no multi-task
dependency system for the task types that don't exist. Ontology-position
tiering isn't a fresh choice made here, it's the only one anything in this
codebase actually supports, reusing the exact `FIELD_TIERS` lookup
`app.context_builder` already established for the same leaf `node_type`s.

## This is what actually fixes the bug class the plan describes

`next_field_to_ask()`'s `timeline` handling required its own explicit
end-of-function check, separate from the main `PROJECT_FIELDS`/`ROOM_FIELDS`
walk (see `ARCHITECTURE_BASELINE.md`'s corrections section — this was
already confirmed to be deliberate, not a bug, when that baseline was
written). The plan's real point survives that correction: the risk isn't
this specific field, it's that a *second, hand-maintained list* of fields
to walk can drift from what the ontology actually defines. `find_knowledge_gap`
has exactly one source of truth — walk `ontology/v1.yaml`'s structure via
each level's fixed field order, check `KnowledgeNode` for a value — so a
field added to the ontology and forgotten in the walk order is now the only
way to reproduce that bug class, not "someone forgot the special case."

## `moreRoomsPending` isn't walked — not a new gap

Phase 3 already decided this field is dialogue mechanics, not a project
fact (`ontology/PHASE3_ONTOLOGY.md` — out of ontology entirely, belongs to
`DialogueState`). `find_knowledge_gap` not asking about it is that decision
holding, not a regression from `next_field_to_ask()`'s behavior.

## `generate_question` is a thin adapter, not new generation logic

`app.question_engine.generate_question(gap, context)` calls the exact same
`deepinfra.generate_question` the live turn already uses — same model,
same prompt, same retry-phrasing behavior — just fed `KnowledgeGap.field_label`
instead of a raw `PartialContext` field-name string. No new model
integration.

## Still not the live cutover

`find_knowledge_gap`/`generate_question` are real and tested, but nothing
calls them from `app/graph.py`. The plan's own Phase 15 exit criteria
("old `PartialContext`-based version deleted") is the live cutover this
whole implementation pass has consistently deferred — see
`ontology/PHASE7_CONTEXT_BUILDER.md`'s Phase 15 dependency note. That gap is
now closed in the sense that matters: the replacement exists and works.
Whether to actually delete `next_field_to_ask()` and rewire `app/graph.py`
is a separate, explicit, higher-risk decision — see the final wrap-up.
