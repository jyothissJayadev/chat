# Prompt Contracts Reference

Single source of truth for prompt engineering on this codebase: every LLM
call in the live pipeline, the **exact structure it's allowed to produce**
(the ontology), the **input format** each prompt receives, and the
**output format** (Pydantic schema) it must return.

Source of truth for each section: `ontology/v1.yaml` (structure),
`app/prompts.py` (prompt text/input templates), `app/llm.py` (output
schemas + model config), `app/canonical_mapper.py` (connection/freeform
rules).

---

## 1. Turn flow (who calls what)

```
User message
   │
   ▼
classify_operations  ─────────────────────────────► [Operation, ...]
   │ (splits message into ops, tags intent, grounds each to a graph path)
   │
   ├─ any write/retrieval op has connection == null?
   │     └─ YES → resolve_room_connections (batched, once) → ask user → STOP (turn ends)
   │
   ▼ (every op now has a connection)
resolve_context_changes  ─────────────────────────► [ContextChangeResult, ...]
   │ (for every CONTEXT_UPDATE/CONTEXT_DELETE op: field values, freeform
   │  entities, or exact deletion paths)
   │
   ▼
graph writes (app/context_builder.py, app/canonical_mapper.py — pure Python,
no LLM: apply_to_graph, map_to_canonical)
   │
   ▼
generate_turn_summary  ────────────────────────────► TurnSummary
   (turns whatever happened this turn — changes, retrieved context, catalog
    results, still-open field — into one reply + next question)
```

Two more calls sit outside this critical path:
`generate_search_keywords` (DATABASE_RETRIEVAL only, runs before
`rag.query_catalog`) and `generate_answer` (DIRECT_ANSWER only, runs
concurrently, streamed, bypasses `generate_turn_summary` entirely).

All structured calls go through `instructor` (`app/llm.py:structured_client`,
mode `TOOLS`) against `settings.model_intent_classifier` /
`model_extraction` — currently both
`accounts/fireworks/models/gpt-oss-120b`, reasoning_effort `low`. A call
gets one attempt at `max_retries=0`; on failure it's retried once at
`max_retries=4`; if that also raises, a salvage pass scans the failed
attempts' raw tool-call/content for a parseable JSON object before giving
up for real.

---

## 2. Fixed structure — the ontology (`ontology/v1.yaml`)

This is the **complete, closed set** of node types and fields a write can
ever target. No prompt, model, or code path may invent a node type or a
canonical-path segment outside this table. `versioning_policy:
additive-only` — entries get added over time, never renamed.

| Node type | Kind | `instantiable` | Fields (leaves) | Children |
|---|---|---|---|---|
| `Project` | root | no | — | BasicInformation, Budget, Timeline, Rooms, Requirements, Quotation, Unmapped |
| `BasicInformation` | singleton | no | `ProjectType` | — |
| `Budget` | singleton | no | `Total` | — |
| `Timeline` | singleton | no | `Value` | — |
| `Rooms` | **instance per room** | yes | `RoomType`, `Budget`, `Style`, `SquareFootage`, `ExistingFurniture` | Materials, Furniture, Attributes, Unmapped |
| `Materials` | **instance per item** | yes | `Label`, `Material`, `Specification` | — |
| `Furniture` | **instance per item** | yes | `Label`, `Material`, `Notes` | — |
| `Attributes` | **instance per fact** | yes | `Label` | — |
| `Requirements` | container | no | — | Constraints, ClientPreferences |
| `Constraints` | **instance per rule** | yes | `Label` | — |
| `ClientPreferences` | **instance per preference** | yes | `Label` | — |
| `Unmapped` | **instance per fact** | yes | `Label` | — (review bucket; not final) |
| `Quotation` | reserved, unused | no | — | RoomLineItems |
| `RoomLineItems` | reserved, unused | yes | `Amount`, `Basis` | — |

**Canonical path shape**: dot-delimited, root at `Project`.
`Project.Rooms.<room_id>.Materials.<item_slug>.Label`.
`<room_id>` = an opaque 8-char hex id (`uuid4().hex[:8]`, minted by the
pipeline). `<item_slug>` = `canonical_mapper.slugify(raw_entity)`
(lowercased, non-alphanumerics → `_`).

**Root-relative form** (what every prompt actually sees/produces, and what
`connection`/`deletion_targets` use): the same path with the leading
`Project.` stripped — e.g. `Rooms.a1b2c3d4.Materials.countertop`, never
`Project.Rooms.a1b2c3d4...`. The literal string `"Project"` (no path after
it) means "the whole project, no room" — used for `projectType`,
`overallBudget`, `timeline`.

**Freeform vocabulary** — the only 5 node types a freeform mention can ever
be classified into (`canonical_mapper._FREEFORM_NODE_TYPES`):
`Materials | Furniture | Attributes | Constraints | ClientPreferences`.
A mention that can't be confidently placed (best-match embedding score
`< 0.75`) lands in `Unmapped` instead — never a made-up type.

**Structured fields** (never routed through freeform matching — always
exact schema slots):

| Field name | Scope | Tier | Canonical path |
|---|---|---|---|
| `projectType` | project | critical | `Project.BasicInformation.ProjectType` |
| `overallBudget` | project | critical | `Project.Budget.Total` |
| `timeline` | project | moderate | `Project.Timeline.Value` |
| `roomType` | room | critical | `Project.Rooms.<id>.RoomType` |
| `budgetOrRequirement` | room | critical | `Project.Rooms.<id>.Budget` |
| `style` | room | moderate | `Project.Rooms.<id>.Style` |
| `squareFootage` | room | moderate | `Project.Rooms.<id>.SquareFootage` |
| `existingFurniture` | room | optional | `Project.Rooms.<id>.ExistingFurniture` |
| `materials` | room | optional | `Project.Rooms.<id>.Materials.<slug>.{Label,Material,Specification}` |

(`app/models.py:FIELD_TIERS`/`FIELD_LABELS`)

---

## 3. The project-tree text format (shared input block)

`context_builder.render_project_tree_text(project_id)` renders the live
graph as plain text and is injected into **every** classifier/resolver
prompt as `CURRENT DATA TREE`. Value-only (no `node_id`/`version`/
`confidence`/`lifecycle`/embedding), root-relative, and a field with no
value is omitted entirely rather than shown blank.

```
Project
├── BasicInformation
│   └── ProjectType = "renovation"
├── Budget
│   └── Total = "25 lakhs"
├── Rooms
│   └── Rooms.a1b2c3d4
│       ├── RoomType = "kitchen"
│       ├── Style = "modern"
│       ├── Materials.countertop
│       │   Label="countertop", Material="granite"
│       └── Furniture.island
│           Label="island"
├── Requirements
│   └── ClientPreferences.no_dark_colors
│       Label="no dark colors"
└── Unmapped
    └── Unmapped.smart_lighting
        Label="smart lighting"
```

An empty project renders as just `"Project"` with no children.

---

## 4. Call-by-call contracts

### 4.1 `classify_operations` — the fork point

**Purpose**: split the raw message into independent operations, tag each
with exactly one intent, and ground each to a spot in the live tree.
Never decides graph mutation mechanics — text/intent/connection only.

**Model**: `model_intent_classifier`, `max_tokens=1024`.

**Input** (`prompts.classify_operations_system`/`_user`):

- System = `CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE` with `{{current_data_tree}}`
  replaced by §3's tree text.
- User message:
  ```
  Conversation so far:
  <history text>

  Currently pending question (if any): <field label | "none">

  Latest message:
  <raw user message>
  ```

**Output** — `OperationClassification`:

```json
{
  "operations": [
    {
      "id": "op_1",
      "text": "cleaned operation text preserving all meaningful user information",
      "intent": "CONTEXT_UPDATE | CONTEXT_DELETE | CONTEXT_RETRIEVAL | DATABASE_RETRIEVAL | DIRECT_ANSWER",
      "connection": "exact canonical graph path | new room name | Project | null"
    }
  ]
}
```

`id` is never trusted from the model — `app.llm.classify_operations`
overwrites it by list order (`op_1`, `op_2`, ...) after parsing.

**`connection` — the 4 legal values, exactly**:

| Value | Meaning | Rule |
|---|---|---|
| exact canonical path (e.g. `Rooms.a1b2c3d4.Materials.countertop`) | targets an existing node | only when that literal path is present in `CURRENT DATA TREE` |
| new room name (e.g. `"Kids Bedroom"`) | a room that doesn't exist yet | only for a room genuinely being introduced |
| `"Project"` | whole-project scope | `projectType`/`overallBudget`/`timeline` only |
| `null` | unresolved | target not found, ambiguous, or no reliable scope — **never guessed** |

Hard rule: an EDIT/DELETE on an entity that can't be found in the tree
must return `null`, **never** fall back to the parent room. A brand-new
item being added to an *existing* room may use that room's path as
`connection`.

**Intent vocabulary** (`OperationIntent`, exactly 5, exhaustive):
`CONTEXT_UPDATE | CONTEXT_DELETE | CONTEXT_RETRIEVAL | DATABASE_RETRIEVAL | DIRECT_ANSWER`

---

### 4.2 `resolve_room_connections` — Room Resolution Agent

**Purpose**: batched, once per turn, over every write/retrieval op whose
`connection` came back `null` from 4.1. Resolves ONLY the room — never
touches material/size/style/price. Always returns a question + ≥1 option
per op (no "confidently resolved, skip asking" verdict exists in this
prompt — the caller always surfaces the question).

**Model**: `model_intent_classifier`, `max_tokens=1024`.

**Input** (`prompts.room_resolution_agent_system`/`_user`):

- System = `ROOM_RESOLUTION_AGENT_SYSTEM_TEMPLATE` with
  `{{current_data_tree}}` → §3 tree text, `{{operations}}` → JSON:
  ```json
  [
    {"text": "add a sofa", "intent": "CONTEXT_UPDATE", "connection": null}
  ]
  ```
- User message: literal `"Return the JSON array now."`

**Output** — `RoomResolutionBatch`:

```json
{
  "resolutions": [
    {
      "text": "add a sofa",
      "intent": "CONTEXT_UPDATE",
      "question": "Where would you like to place the sofa?",
      "options": [
        {"id": "r2", "label": "Living Room"}
      ]
    }
  ]
}
```

Rules: one result per input op, same order (positional matching, no id
round-trip). `id` is the **exact** existing room id when the option is a
real room in the tree; `id: null` for an inferred/not-yet-existing room —
never invent a room id. `options` capped at 3 when ambiguous; exactly 1
when confidently resolved. Labels are plain room names only — never a
canonical path, node id, or explanation.

---

### 4.3 `resolve_context_changes` — field/entity/deletion extraction

**Purpose**: ONE combined call over every `CONTEXT_UPDATE`/`CONTEXT_DELETE`
op in the turn (every op already carries a resolved `connection` from 4.1/
4.2 — this call never re-derives room identity). Per op: which structured
fields it states, which freeform entities it mentions, or (for deletes)
which exact existing paths it removes.

**Model**: `model_extraction`, `max_tokens=2048`.

**Input** (`prompts.resolve_context_changes_system`/`_user`):

- System = `RESOLVE_CONTEXT_CHANGES_SYSTEM_TEMPLATE` with
  `{{current_data_tree}}` → §3 tree text, `{{operations}}` → JSON:
  ```json
  [
    {"text": "make the kitchen modern", "intent": "CONTEXT_UPDATE", "connection": "Rooms.a1b2c3d4"}
  ]
  ```
- User message: literal `"Return the JSON object now."`

**Output** — `ContextChangeBatch`:

```json
{
  "results": [
    {
      "text": "make the kitchen modern",
      "intent": "CONTEXT_UPDATE",
      "fields": {
        "projectType": null, "overallBudget": null, "timeline": null,
        "budgetOrRequirement": null, "style": "modern", "squareFootage": null,
        "existingFurniture": null, "materials": null
      },
      "freeform_entities": [
        {"raw_entity": "walnut TV cabinet", "node_type_hint": "Furniture"}
      ],
      "deletion_targets": []
    }
  ]
}
```

Field-level rules:
- `text`/`intent` copied back verbatim from the input op.
- `projectType`/`overallBudget`/`timeline` set **only** when `connection == "Project"`.
- `budgetOrRequirement`/`style`/`squareFootage`/`existingFurniture`/`materials` set **only** for a room-scoped op.
- A hedged/approximate statement ("maybe around 4 lakh") still counts as stated — keep the hedge wording, don't null it.
- `freeform_entities`: anything mentioned that isn't a structured field slot — one entry per distinct entity, `raw_entity` in the user's own wording, `node_type_hint` is one of the 5 freeform types from §2 or `null` (inferred downstream by `canonical_mapper.map_to_canonical`, embedding threshold `0.75`).
- `deletion_targets` (CONTEXT_DELETE only): exact canonical paths **copied character-for-character** from `CURRENT DATA TREE` — never invented. Deleting a room's own path retracts everything under it. Empty list if nothing confidently matches — never guess.
- `fields`/`freeform_entities` must be empty/null on a DELETE result; `deletion_targets` must be empty on an UPDATE result.

`MaterialChange` shape inside `fields.materials`:
```json
{"item": "countertop", "material": "granite", "specification": "polished, 2cm"}
```
(`specification` optional, everything else required when present.)

---

### 4.4 `generate_search_keywords`

**Purpose**: DATABASE_RETRIEVAL only — turns a catalog-search request into
a short vector-search query string.

**Model**: `model_question_gen`, plain chat completion (not structured),
`temperature=0.2`, `max_tokens=64`.

**Input**: system = `SEARCH_KEYWORDS_SYSTEM` (static). User = `f"Request: {query}"`.

**Output**: raw string, no JSON — "Output ONLY the search string, nothing else."

---

### 4.5 `generate_turn_summary` — the final join step

**Purpose**: the ONE call that turns whatever pieces actually happened
this turn into the reply text.

**Model**: `model_question_gen`, `max_tokens=768`, `max_retries=2`.

**Input** (`prompts.turn_summary_user`): system = `TURN_SUMMARY_SYSTEM`
(static). User = `f"This turn's pieces: {pieces}"` where `pieces` is:

```json
{
  "changes": [
    {"path": "Project.Rooms.a1b2c3d4.Style", "before": null, "after": "modern", "action": "created"}
  ],
  "context_retrieval": [
    {"field": "budgetOrRequirement", "value": "25 lakhs"}
  ],
  "database_results": [
    {"title": "Walnut Veneer Panel", "description": "..."}
  ],
  "pending_gap": ["square footage", "existing furniture"]
}
```
`action` ∈ `created | updated | deleted | deleted_room`. Any piece that
didn't happen this turn is `[]`/`null`, not omitted.

**Output** — `TurnSummary`:

```json
{
  "database_summary": "string or null — null if no catalog search happened",
  "context_summary": "string or null — null if no retrieval happened",
  "changes_summary": "string or null — null if nothing changed",
  "next_message": "the next question to ask, or a closing line if nothing is left",
  "is_question": true
}
```

Rules baked into the system prompt: one natural reply, never a bulleted
list or labeled sections, never repeats a piece verbatim. `is_question`
true only if `next_message` is an open question awaiting an answer; if
`pending_gap` is non-empty the reply must end with ONE combined question
covering it. If literally nothing happened this turn, say so briefly and
set `is_question: false`.

---

### 4.6 `infer_missing_field` (not on the live write path today)

**Purpose**: best-guess a single field value when a user declines to
answer. Wired and tested but not called from the live turn (declines are
handled as a whole-room skip instead — see `app.graph._is_decline`).

**Model**: `model_extraction`, plain chat completion, `temperature=0.3`, `max_tokens=64`.

**Input**: system = `INFER_MISSING_FIELD_SYSTEM` (static). User:
`f"Known project details: {context}\n\nField to estimate: {field_name}"`.

**Output**: raw short string — "a typical/reasonable figure... No
explanation, no caveats, just the value itself" (e.g. `"150 sqft"`).

---

### 4.7 `generate_answer` — DIRECT_ANSWER, streamed

**Purpose**: answers a general question directly; runs concurrently with
everything else, bypasses `generate_turn_summary`.

**Model**: `model_answer`, streamed chat completion, no schema.

**Input** (`prompts.generate_answer_user`): system = `GENERATE_ANSWER_SYSTEM`
(static). User:
```
Project context: {known_fields dict}

Retrieved references: <text | "none">

Conversation so far:
<history>

User: <message>
```

**Output**: free-text token stream, no structure.

---

## 5. Values NOT allowed anywhere

- A `connection`/`deletion_targets` path outside §2's ontology tree, or one
  that doesn't literally appear in the current `CURRENT DATA TREE` block.
- A freeform `node_type_hint` outside the 5-type list in §2.
- A structured field set from a DIFFERENT operation's text, or from
  `CURRENT DATA TREE` itself rather than the current message.
- Guessing a `connection`/deletion target when ambiguous — must be `null`
  / omitted instead.
- Any output field not present in that call's schema (all prompts end
  with "No markdown. No explanations. No additional fields.").

## 6. Fixed vocab quick-reference

| Enum | Values |
|---|---|
| `OperationIntent` | `CONTEXT_UPDATE`, `CONTEXT_DELETE`, `CONTEXT_RETRIEVAL`, `DATABASE_RETRIEVAL`, `DIRECT_ANSWER` |
| `TaskType` (1:1 with intents) | `EDIT_CONTEXT`, `DELETE_CONTEXT`, `RETRIEVE_CONTEXT`, `DATABASE_QUERY`, `ANSWER` |
| Freeform `node_type_hint` | `Materials`, `Furniture`, `Attributes`, `Constraints`, `ClientPreferences`, `null` |
| `FieldStatus` | `confirmed`, `assumed`, `skipped` |
| `ChangedBy` | `user_message`, `inferred`, `system_default` |
| Field tier | `critical`, `moderate`, `optional` |
