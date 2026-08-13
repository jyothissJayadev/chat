# Classify Intent System - Complete Process Documentation

## Overview

The classify intent system is the **entry point** for every user message in the chat pipeline. It replaces the legacy `classify_intent` (single/rarely-double label per message) with a **multi-operation, structured-output classifier** called `classify_operations` that:

1. **Splits** the user's message into independent meaningful operations
2. **Classifies** each operation with exactly one of 5 intent labels
3. **Grounds** each operation to a spot in the project's live graph (`connection`)

This all happens in **one LLM call** with structured output (Pydantic models via Instructor).

---

## Architecture Flow

```
User Message → ChatSession → classify_intent_node (Graph Entry Point)
                                    ↓
                          understand() [app/understanding.py]
                                    ↓
                          classify_operations() [app/llm.py]
                                    ↓
                          LLM Call (accounts/fireworks/models/gpt-oss-120b)
                                    ↓
                          OperationClassification → TaskSpec[]
                                    ↓
                          classify_intent_node returns {intent, tasks, pending_operation_questions?, trace}
                                    ↓
                          _route_intent() routes to downstream nodes
```

---

## Core Components

### 1. Graph Entry Point: `classify_intent_node` (`app/graph.py:178`)

**Purpose**: Thin wrapper around `app.understanding.understand()` that handles:
- LLM prompt/response capture for tracing (Langfuse)
- Resume path for clarifying questions (`pending_operation_questions`)
- Connection grounding validation → generates clarifying questions if needed
- Returns routing state (`intent`, `tasks`, `pending_operation_questions`)

**Key Logic** (lines 178-262):

```python
async def classify_intent_node(state: GraphState) -> dict:
    # 1. Check for RESUME path (user answered clarifying questions from previous turn)
    pending = state.get("pending_operation_questions")
    if pending:
        answers = state.get("operation_answers") or {}
        tasks = [TaskSpec(**t) for t in pending["tasks"]]
        for task in tasks:
            if task.op_id in answers:
                task.connection = answers[task.op_id]  # Merge user's answer
        raw_intents = [TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
        output_summary = f"resumed {len(tasks)} operation(s), {len(answers)} answer(s) applied"
    
    # 2. FRESH CLASSIFICATION path
    else:
        tree_text = await context_builder.render_project_tree_text(state["project_id"])
        meaning = await understanding.understand(
            state["message"],
            state["history"],
            pending_field=(state.get("pending_gap") or {}).get("canonical_path"),
            tree_text=tree_text,
            capture=capture,  # Captures LLM messages + raw output for trace
        )
        tasks = meaning.tasks
        raw_intents = [TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
        output_summary = ",".join(raw_intents)
        if raw_intents != meaning.raw_intents:
            output_summary += f" (guard adjusted from: {','.join(meaning.raw_intents)})"
    
    # 3. Trace the LLM call
    entry = _trace("classify_intent", model, message, output_summary, start, ...)
    
    # 4. CONNECTION VALIDATION & CLARIFYING QUESTIONS
    # Only for multi-op turns (or resume) — single op uses downstream resolve_context
    unresolved = _unresolved_connection_tasks(tasks) if (pending or len(tasks) > 1) else []
    if unresolved:
        # LLM-judged, not deterministic: one generate_clarification_question() call per
        # unresolved op (concurrent), each deciding needs_clarification for itself — see
        # app/llm.py and app/prompts.py:GENERATE_CLARIFICATION_QUESTION_SYSTEM_TEMPLATE.
        questions = await _generate_operation_questions(state["project_id"], state["message"], unresolved)
        # Only hold the batch back if the model actually flagged real ambiguity — an op
        # judged needs_clarification=False keeps connection=None and falls through to the
        # ordinary single-op resolve path below instead of blocking the turn.
        if questions:
            return {
                "intent": raw_intents,
                "tasks": tasks,
                "pending_operation_questions": {"tasks": [t.model_dump() for t in tasks], "questions": questions},
                "question_generated": True,
                "trace": [entry, clarify_entry],
            }
        return {"intent": raw_intents, "tasks": tasks, "pending_operation_questions": None, "trace": [entry, clarify_entry]}

    return {"intent": raw_intents, "tasks": tasks, "pending_operation_questions": None, "trace": [entry]}
```

**Routing Decision** (`_route_intent`, lines 1007-1042):
- Reads `state["tasks"]` directly (not `state["intent"]`)
- Multi-label (`len(tasks) > 1`) → `handle_split_intents`
- Single task → routes by `task.type` to:
  - `RETRIEVE_CONTEXT` → `retrieve_context`
  - `DIRECT_ANSWER` → `generate_answer`
  - `DATABASE_QUERY` → `query_catalog`
  - `EDIT_CONTEXT` → `build_context`
  - `DELETE_CONTEXT` → `delete_context`
- `pending_operation_questions` present → `analyze_context` (short-circuits)

---

### 2. Pure Understanding Layer: `understand()` (`app/understanding.py:78`)

**Purpose**: DB-free, side-effect-free function. Input: message + history + tree_text. Output: `Meaning` (tasks + raw_intents).

```python
async def understand(
    message: str,
    history: str = "",
    pending_field: str | None = None,
    *,
    tree_text: str = "Project",
    capture: dict | None = None,
) -> Meaning:
    operations = await llm.classify_operations(
        message, history, pending_field=_pending_field_label(pending_field), tree_text=tree_text, capture=capture
    )
    
    # RULE 15: Empty/unclear input → fallback to DIRECT_ANSWER over full message
    if not operations:
        operations = [llm.Operation(id="op_1", text=message, intent="DIRECT_ANSWER")]
    
    tasks = _operations_to_tasks(operations)
    raw_intents = [op.intent for op in operations]
    return Meaning(tasks=tasks, raw_message=message, raw_intents=raw_intents)
```

**Key Mapping** (`_INTENT_TO_TASK_TYPE`, lines 24-31):
| classify_operations Intent | TaskType |
|---|---|
| CONTEXT_UPDATE | EDIT_CONTEXT |
| CONTEXT_DELETE | DELETE_CONTEXT |
| CONTEXT_RETRIEVAL | RETRIEVE_CONTEXT |
| DATABASE_RETRIEVAL | DATABASE_QUERY |
| DIRECT_ANSWER | ANSWER |

**Operation → TaskSpec** (`_operations_to_tasks`, lines 51-75):
```python
TaskSpec(
    type=_INTENT_TO_TASK_TYPE[op.intent],
    op_id=op.id,           # "op_1", "op_2", ...
    target=op.text,        # Operation's own text
    connection=op.connection,  # Grounding from classifier
)
```

---

### 3. LLM Classifier: `classify_operations()` (`app/llm.py:151`)

**Model**: `settings.model_intent_classifier` (accounts/fireworks/models/gpt-oss-120b)

**Structured Output**: `OperationClassification` → list of `Operation`

```python
class Operation(BaseModel):
    text: str                    # Cleaned operation text preserving user's wording
    intent: OperationIntent      # One of 5 intents
    connection: Optional[str] = None  # Grounding: canonical path | new room name | "Project" | null
    id: str = ""                 # Assigned post-parsing by list order ("op_1", "op_2", ...)

class OperationClassification(BaseModel):
    operations: list[Operation] = Field(default_factory=list)
```

**Retry Strategy** (`_classify_operations_retrying`, lines 172-189):
- Attempt 1: `max_retries=0` (single attempt)
- On `InstructorRetryException`: salvage from failed attempts (JSON in tool_calls or content)
- Attempt 2: `max_retries=4` (full budget) if salvage fails

**Prompt Construction** (`_classify_operations_raw`, lines 192-215):
```python
messages = [
    {"role": "system", "content": prompts.classify_operations_system(tree_text)},
    {"role": "user", "content": prompts.classify_operations_user(message, history, pending_field)},
]
result, completion = await structured_client.chat.completions.create_with_completion(
    model=settings.model_intent_classifier,
    response_model=OperationClassification,
    max_tokens=1024,
    max_retries=max_retries,
    messages=messages,
)
```

---

### 4. Prompt Engineering: `CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE` (`app/prompts.py:34-395`)

**Deliberately verbose** (~360 lines) because this is the **fork point** for all downstream branches — a misclassification here cannot be corrected later.

**Template Structure**:
1. **Role Definition** — Intent, operation-splitting, graph-connection classifier
2. **CURRENT DATA TREE** — Injected via `{{current_data_tree}}` placeholder
3. **CONNECTION Rules** (61-159) — 4 connection types with examples:
   - Exact canonical graph path (existing entity)
   - New room name (normalized)
   - "Project" (project-wide)
   - null (ambiguous/unresolvable)
4. **INTENTS** (161-203) — 5 intents with examples:
   - CONTEXT_UPDATE: New info, preferences, changes to save
   - CONTEXT_DELETE: Explicit removal/retraction
   - CONTEXT_RETRIEVAL: Questions about stored project info
   - DATABASE_RETRIEVAL: Catalog/product search
   - DIRECT_ANSWER: General knowledge questions
5. **CORE RULES** (204-374) — 19 rules covering:
   - Never lose information (Rule 1)
   - Split by meaning, not punctuation (Rule 2)
   - One operation = one intent (Rule 3)
   - Don't over-split (Rule 4)
   - Preserve user details (Rule 5)
   - Resolve references using message + tree (Rule 6)
   - Connection must match operation type (Rule 7)
   - No parent fallback for edit/delete (Rule 8)
   - Existing vs new determination (Rule 9)
   - Context vs Database distinction (Rule 10)
   - Update vs Delete (Rule 11)
   - Multi-room/entity splitting (Rule 12)
   - Mixed operations preservation (Rule 13)
   - Order preservation (Rule 14)
   - Spelling normalization (Rule 15)
   - Don't invent information (Rule 16)
   - No graph mutation reasoning (Rule 17)
   - Final completeness check (Rule 18)
   - Empty input handling (Rule 19)
6. **OUTPUT Schema** (380-394) — JSON with operations array

**User Prompt** (`classify_operations_user`, lines 402-407):
```
Conversation so far: {history}
Currently pending question (if any): {pending_field or 'none'}
Latest message: {message}
```

---

## Connection Grounding & Clarifying Questions

### Connection Types (from prompt)

| Connection Value | When Used |
|---|---|
| `Rooms.a1b2c3d4.Materials.countertop` | Existing entity in CURRENT DATA TREE |
| `Kids Bedroom` | New room (normalized name) |
| `Project` | Project-wide info (budget, style, type) |
| `null` | Ambiguous, missing target for edit/delete, unresolvable |

### Clarifying Question Flow

1. **Trigger**: `classify_intent_node` finds tasks with `connection=None` AND (multi-op OR resume)
2. **Judge + generate**: `_generate_operation_questions(project_id, message, unresolved_tasks)` calls
   `llm.generate_clarification_question(message, task.target, intent, tree_text)` once per unresolved
   task (concurrently, via `asyncio.gather`) — a real LLM call per op, not a deterministic room-list
   builder. Each call returns `{needs_clarification, question, options}` (`app.llm.ClarificationQuestion`),
   using the verbatim prompt in `app/prompts.py:GENERATE_CLARIFICATION_QUESTION_SYSTEM_TEMPLATE`. Only
   ops the model flags `needs_clarification=true` (with a non-null `question`) become entries in the
   returned list — each `{op_id, text, question, options: [{id, label}, ...], allow_custom: true}`.
   An op the model judges resolvable on its own (`needs_clarification=false`) is simply omitted; its
   `connection` stays `None` and it falls through to the ordinary single-op resolve path.
3. **Hold or fall through**: if the returned question list is non-empty, `classify_intent_node` holds
   the *whole* batch — resolved and unresolved tasks alike — via `pending_operation_questions`. If
   every unresolved op turned out `needs_clarification=false`, the list is empty and the turn proceeds
   to normal routing exactly as if there had been no unresolved connections at all.
4. **Return**: `pending_operation_questions` with serialized tasks + questions (only when non-empty)
5. **Route**: `_route_intent` sees `pending_operation_questions` → `analyze_context` (short-circuit)
6. **Next Turn**: User answers via `ChatRequest.operation_answers` (dict: `op_id` → chosen option label
   or freely typed text — the viewer renders both the option list and a text input per question)
7. **Resume**: `classify_intent_node` reloads tasks, merges answers into `connection`, re-validates
8. **Partial Answer**: Only still-unresolved ops get re-asked (via the same LLM-judged cycle above);
   answered ones stay answered. This repeats — one clarification round trip per turn — until every
   `connection: null` in the batch is gone, only then do the tasks proceed to execution.

---

## Data Structures

### GraphState (relevant fields)
```python
class GraphState(TypedDict):
    message: str
    history: str
    project_id: str
    intent: list[str]              # Telemetry only (raw_intents)
    tasks: list[TaskSpec]          # PRIMARY routing key
    pending_operation_questions: Optional[dict]  # {tasks: [...], questions: [...]}
    operation_answers: Optional[dict]  # {op_id: connection_value} from user
    trace: list[TraceEntry]
    # ... other fields
```

### TaskSpec (`app/tasks.py:31`)
```python
class TaskSpec(BaseModel):
    type: TaskType           # EDIT_CONTEXT, DELETE_CONTEXT, RETRIEVE_CONTEXT, DATABASE_QUERY, ANSWER
    op_id: str               # "op_1", "op_2", ... (matches classify_operations output)
    target: Optional[str]    # Operation's own text
    room_hint: Optional[str] # Derived from connection downstream (NOT set here)
    connection: Optional[str] # Classifier's grounding guess (canonical path | room name | "Project" | None)
```

### Operation (`app/llm.py:89`)
```python
class Operation(BaseModel):
    text: str
    intent: Literal["CONTEXT_UPDATE", "CONTEXT_DELETE", "CONTEXT_RETRIEVAL", "DATABASE_RETRIEVAL", "DIRECT_ANSWER"]
    connection: Optional[str] = None
    id: str = ""  # Assigned post-parsing
```

---

## Integration with Rest of Codebase

### Downstream Nodes (routed from `classify_intent`)

| TaskType | Node | Purpose |
|---|---|---|
| RETRIEVE_CONTEXT | `retrieve_context_node` | Query KnowledgeNode graph for stored project info |
| DIRECT_ANSWER | `generate_answer_node` | Answer general questions (no project/db access) |
| DATABASE_QUERY | `query_catalog_node` | Search product catalog (RAG) |
| EDIT_CONTEXT | `build_context_node` | Extract fields + write to KnowledgeNode graph |
| DELETE_CONTEXT | `delete_context_node` | Retract nodes from KnowledgeNode graph |
| (multiple) | `handle_split_intents_node` | Run multiple operations concurrently via `asyncio.gather` |

### Multi-Operation Execution (`handle_split_intents_node`, `app/graph.py:711`)

```python
async def handle_split_intents_node(state: GraphState) -> dict:
    tasks = state["tasks"]
    results = await asyncio.gather(*[
        _execute_single_task(state, task) for task in tasks
    ])
    # Merge results: trace, retrieved, written, confirmations, questions, etc.
    return merged_result
```

Each task runs through its appropriate node with `task.target` as the message (per-operation text).

### Context Building (`build_context_node`, `app/graph.py:339`)

- Receives `TaskSpec` with `connection` as anchor hint
- `app.execution._resolve_write_task` uses `connection` via `canonical_mapper.split_connection` → `resolve_context`
- `connection` is **never a final write path** — only an anchor hint for resolution

### Clarifying Questions Storage (`ChatSession`, `app/models.py:146-147`)

```python
pending_operation_questions: Optional[dict] = None
# Structure: {"tasks": [TaskSpec...], "questions": [str...]}
```

Round-tripped via `ChatRequest.operation_answers` → `state["operation_answers"]` → merged in resume branch.

---

## Key Design Decisions

### 1. Single LLM Call for Split + Classify + Ground
- **Before**: `classify_intent` (single label) + regex guards + `segment_clauses` (room splitter) — 3+ passes
- **After**: One structured call → `OperationClassification` with all 3 concerns

### 2. Connection as Anchor Hint, Not Final Path
- Classifier outputs `connection` grounded to CURRENT DATA TREE
- Downstream `canonical_mapper.is_grounded_connection()` / `room_id_from_connection()` validate
- `resolve_context()` does fuzzy matching / creation — classifier doesn't decide graph mutations

### 3. Multi-Label Routing via `state["tasks"]`
- `state["intent"]` is telemetry only (list of raw intent strings)
- Routing reads `state["tasks"]` directly — each task has its own `type`
- Enables true parallel execution in `handle_split_intents_node`

### 4. Clarifying Questions Only for Multi-Op / Resume
- Single operation: downstream `resolve_context` handles ambiguous connections
- Multi-op: connection is critical disambiguator across operations sharing one project
- Resume: already passed the gate, always re-validates

### 5. RULE 15 Fallback (Empty Operations)
- If classifier returns empty list → wrap full message as `DIRECT_ANSWER`
- Guarantees every turn produces a reply (e.g., "hello" → answer, not silence)

---

## Tracing & Observability

Every `classify_intent` turn produces a `TraceEntry` (`app/models.py:20`):
```python
TraceEntry(
    node_name="classify_intent",
    model_used=settings.model_intent_classifier,
    input_summary="user message",
    output_summary="CONTEXT_UPDATE,CONTEXT_RETRIEVAL (guard adjusted from: ...)",
    duration_ms=...,
    llm_input=[{"role": "system", ...}, {"role": "user", ...}],  # Full prompt
    llm_output='{"operations": [...]}',  # Raw JSON from model
)
```

Captured via `capture` dict threaded through `understand()` → `classify_operations()` → `_classify_operations_raw()`.

---

## Testing

Key test files:
- `tests/test_graph.py` — `classify_intent_node` behavior (resume, clarifying questions, routing)
- `tests/test_understanding.py` — `understand()` pure function tests
- `tests/test_llm.py` — `classify_operations` system prompt coverage, LLM integration

Example test (`test_graph.py:231`):
```python
async def test_classify_intent_node_resumes_from_pending_operation_questions_without_reclassifying():
    # State with pending_operation_questions from previous turn
    # operation_answers provided
    # Assert: no LLM call, tasks reloaded, connections merged, unresolved re-asked
```

---

## Summary

The classify intent system is a **pure, testable, single-call classifier** that:
1. **Splits** messages into independent operations
2. **Classifies** each with 1 of 5 intents (mechanically mapped to TaskType)
3. **Grounds** each to the project graph via `connection`
4. **Validates** connections → generates clarifying questions only when needed
5. **Routes** via `state["tasks"]` to specialized downstream nodes
6. **Resumes** seamlessly from clarifying-question answers

All without touching the database — the graph query (`render_project_tree_text`) happens in `classify_intent_node` before calling `understand()`, keeping the understanding layer pure.