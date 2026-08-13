# CHAT_EXPLANATION

Complete flow analysis of the Interior Design Chat application — how a user
message travels through the system, and exactly **who controls what happens
after `classify_intent`**.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & HTTP/SSE Flow](#2-entry-point--http-sse-flow)
3. [The Orchestrator: `build_graph()` (the answer to "who controls the action")](#3-the-orchestrator-build_graph-the-answer-to-who-controls-the-action)
4. [The `GraphState` Contract](#4-the-graphstate-contract)
5. [`classify_intent_node` in Detail](#5-classify_intent_node-in-detail)
6. [The Understanding Layer (what `classify_intent` actually calls)](#6-the-understanding-layer-what-classify_intent-actually-calls)
7. [Every Downstream Node, Explained](#7-every-downstream-node-explained)
8. [Multi-Operation Batch Execution (`app/execution.py`)](#8-multi-operation-batch-execution-appexecutionpy)
9. [State Persistence: Mongo (sessions) + Neo4j (knowledge graph)](#9-state-persistence-mongo-sessions--neo4j-knowledge-graph)
10. [Complete End-to-End Walkthrough Scenarios](#10-complete-end-to-end-walkthrough-scenarios)
11. [Observability, SSE Events & the Viewer](#11-observability-sse-events--the-viewer)
12. [Key File Map & Line References](#12-key-file-map--line-references)

---

## 1. High-Level Architecture

```
 User browser (viewer.html)
        │  POST /chat  (SSE streaming)
        ▼
 FastAPI app (app/main.py)
        │  app/chat.py  ── run_chat_turn()
        │
        ▼
 LangGraph StateGraph  (app/graph.py, built by build_graph())
        │  app_graph.astream(state, stream_mode=[...])
        │
        ├──▲ entry point: "classify_intent"  →  classify_intent_node
        │  │
        │  │  └► app/understanding.understand()  →  app/llm.classify_operations()  (LLM #1)
        │  │       produces a list of TaskSpec (operations + intents + connections)
        │  │
        │  │  └► if any write task has connection=None  → generate clarifying questions
        │  │       (app/llm.resolve_room_connections(), LLM #2) and HOLD the turn
        │  │
        │  ▼
  conditional edge  _route_intent(state)  ──►  one of:
        │          CONTEXT_RETRIEVAL → retrieve_context
        │          DIRECT_ANSWER     → generate_answer
        │          DATABASE_RETRIEVAL→ query_catalog
        │          CONTEXT_UPDATE    → build_context
        │          CONTEXT_DELETE    → delete_context
        │          decline_field     → decline_field
        │          confirm_conflict  → confirm_conflict
        │          split             → handle_split_intents
        │          operation_questions_pending → analyze_context  (short-circuit)
        │
        ▼
  ...more nodes, each with conditional/static edges (see §7) ...
        │
        ▼
  analyze_context ──► END
        │
        ▼
 run_chat_turn persists ChatSession + KnowledgeNodes and yields SSE events
```

**The stack (how each piece stores/executes):**

| Concern | Technology | Where |
|---|---|---|
| HTTP + server-sent events | FastAPI | `app/main.py`, `app/chat.py` |
| Turn orchestration (the "brain") | **LangGraph `StateGraph`** | `app/graph.py` |
| LLM structured-output calls | `instructor` over an OpenAI-compatible `AsyncOpenAI` client (Fireworks AI) | `app/llm.py`, `app/config.py` |
| Intent/operation classification | one structured LLM call `classify_operations` | `app/llm.py`, `app/understanding.py` |
| Chat session/dialogue state | MongoDB via Beanie (`ChatSession`) | `app/models.py`, `app/database.py` |
| Project knowledge graph | Neo4j (`KnowledgeNode` + versions) | `app/graph_store.py`, `app/neo4j_db.py` |
| Product catalog (semantic search) | MongoDB `$vectorSearch` | `app/rag.py`, `app/models.py` |
| Observability | Langfuse spans + per-node `TraceEntry` log | `app/observability.py`, `app/graph.py` |

---

## 2. Entry Point & HTTP/SSE Flow

### 2.1 FastAPI app — `app/main.py`

```python
app = FastAPI(title="Interior Design Chat", lifespan=lifespan)
app.include_router(chat_router)      # POST /chat
app.include_router(routes_router)    # other REST routes
```

On startup the lifespan handler opens the Mongo and Neo4j connections. The
single live chat endpoint lives in `app/chat.py`.

### 2.2 `POST /chat` — request and streaming

```python
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    image_url: Optional[str] = None
    # {op_id: chosen_value} answering a prior turn's operation_questions —
    # the only structured (non-message-text) reply channel this endpoint has.
    operation_answers: Optional[dict[str, str]] = None


@router.post("/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in run_chat_turn(req.session_id, req.message, req.image_url, req.operation_answers):
            if event["type"] == "heartbeat":
                yield ": keep-alive\n\n"
                continue
            payload = {k: v for k, v in event.items() if k != "type"}
            yield _sse(event["type"], payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

`_sse` wraps each event as `data: {json}\n\n`. Heartbeats are emitted as SSE
comment lines (`: keep-alive`) so a long silent LLM stretch doesn't get dropped
by a proxy idle timeout (8s heartbeat, `HEARTBEAT_SECONDS`).

### 2.3 `run_chat_turn` — one full turn, transport-agnostic

This is the **outer loop**. It builds the initial `GraphState`, hands it to the
compiled LangGraph graph, converts the graph's stream output into SSE events,
and persists the session at the end.

```python
async def run_chat_turn(session_id, message, image_url=None, operation_answers=None):
    session = await get_or_create_session(session_id)
    history = "\n".join(f"{m.role}: {m.content}" for m in session.messages[-10:])
    graph_message = message

    with langfuse.start_as_current_observation(name="chat_turn", as_type="span", ...) as root_span:
        if image_url:
            entry, description = await describe_image_node({"message": graph_message}, image_url)
            session.trace.append(entry)
            yield {"type": "trace", "node": "describe_image", "entries": [...]}
            graph_message = f"{graph_message}\n\n[Image description: {description}]"
            yield {"type": "image_description", "content": description}

        state = {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "message": graph_message,
            "history": history,
            "active_room_id": session.active_room_id,
            "field_attempts": dict(session.field_attempts),
            "intent": [],
            "tasks": [],
            "retrieved": "",
            "pending_question": None,
            "pending_gap": session.pending_gap,                       # round-tripped
            "pending_confirmation": session.pending_confirmation,     # round-tripped
            "pending_operation_questions": session.pending_operation_questions,  # round-tripped
            "operation_answers": operation_answers,                   # this turn's structured reply
            "update_summary": None,
            "answer": "",
            "complete": False,
            "needs_answer": False,
            "question_generated": False,
            "wrapup_message": None,
            "trace": [],
        }

        queue: asyncio.Queue = asyncio.Queue()

        async def run_graph():
            try:
                async for mode, chunk in app_graph.astream(state, stream_mode=["custom", "updates", "values"]):
                    await queue.put((mode, chunk))
            except Exception as exc:
                logger.exception("graph execution failed ...")
                await queue.put(("error", str(exc)))
            finally:
                await queue.put((_SENTINEL, None))

        task = asyncio.create_task(run_graph())
        # drain queue → yield heartbeat/progress/trace/token/operation_progress events
        # ... (event fan-out, see §11)

        # Persist the session:
        session.messages.append(Message(role="user", content=message))
        if final_state["answer"]:
            session.messages.append(Message(role="assistant", content=final_state["answer"]))
        session.active_room_id = final_state.get("active_room_id", session.active_room_id)
        session.field_attempts = final_state.get("field_attempts", session.field_attempts)
        session.trace.extend(final_state["trace"])
        session.status = "complete" if final_state.get("complete") else "in_progress"
        # ... context_updated / wrapup / confirm_change / operation_questions /
        #     ask_question events, then save_session(session)
        yield {"type": "done", "session_id": session.session_id, "status": session.status}
```

Key points:

- **The graph is re-entered fresh every turn** with a state seeded from the
  persisted `ChatSession`. Anything that must survive across turns
  (`pending_question`, `pending_gap`, `pending_confirmation`,
  `pending_operation_questions`, `field_attempts`, `active_room_id`) is
  round-tripped in and back out.
- The graph runs as a background `asyncio.Task`; the streaming loop drains an
  `asyncio.Queue` so stdout work produces SSE without blocking the graph.
- `stream_mode=["custom", "updates", "values"]` gives us token-level custom
  events, per-node update events, and the final full state.

---

## 3. The Orchestrator: `build_graph()` (the answer to "who controls the action")

### The direct answer

> The thing controlling the action **after `classify_intent` is the LangGraph
> `StateGraph` defined by `build_graph()` in `app/graph.py`, working together
> with the conditional-edge router function `_route_intent`.** It is *not*
> `classify_intent_node` calling the next node directly.

There are two layers of control, and it's important to keep them straight:

1. **`build_graph()` — design-time wiring.** It declares *every possible*
   node and *every possible* edge. It is compiled once at import time into the
   global `app_graph = build_graph()` (`app/graph.py:1223`). The compiled graph
   is the object `run_chat_turn` feeds state into via `astream`.
2. **`_route_intent()` — run-time decision.** After `classify_intent` finishes
   and returns, LangGraph's runtime *automatically* invokes `_route_intent`
   (because `build_graph` registered it as the conditional-edge router on the
   `"classify_intent"` node). That function reads the updated state
   (`pending_confirmation`, `pending_operation_questions`, `state["tasks"]`,
   `state["message"]`) and returns the NAME of the next node. LangGraph then
   executes that node. This repeats after **every** node, following whatever
   edge was registered for it.

So: **`classify_intent_node` never calls another node.** It returns a partial
state dict; LangGraph calls `_route_intent`; `_route_intent` picks the branch;
LangGraph runs the branch. The same pattern governs every other junction in
the graph (each `add_conditional_edges` plus its router function).

There is one *exception* to "classify_intent proceeds to normal routing": when
the classifier couldn't ground a write operation, `classify_intent_node`
returns `pending_operation_questions` and `_route_intent` short-circuits to
`analyze_context` (see §5.3). Even then it is still `_route_intent` that makes
the choice — the node merely sets a state flag that the router honors.

### 3.1 The complete wiring (verbatim from `app/graph.py:1056-1223`)

```python
def _route_intent(state: GraphState) -> str:
    """Routes on state["tasks"] (see app.understanding.understand()), not the
    legacy state["intent"] strings — those kept for telemetry."""
    if state.get("pending_confirmation"):
        return "confirm_conflict"
    if state.get("pending_operation_questions"):
        return "operation_questions_pending"
    tasks = state["tasks"] or [TaskSpec(type=TaskType.ANSWER)]
    task_types = [t.type for t in tasks]
    if (
        TaskType.EDIT_CONTEXT not in task_types
        and TaskType.DELETE_CONTEXT not in task_types
        and state.get("pending_gap")
        and _is_decline(state["message"])
    ):
        return "decline_field"
    if len(tasks) > 1:
        return "split"
    return understanding.TASK_TYPE_TO_INTENT[task_types[0]]


def _route_decline(state: GraphState) -> str:
    return "rephrasing" if state.get("pending_question") else "continue"


def _route_after_build_context(state: GraphState) -> str:
    return "confirmation_pending" if state.get("pending_confirmation") else "validate"


def _route_after_delete_context(state: GraphState) -> str:
    return "confirmation_pending" if state.get("question_generated") else "validate"


def _route_completeness(state: GraphState) -> str:
    return "complete" if state["complete"] else "incomplete"


def _route_after_split(state: GraphState) -> str:
    if state.get("pending_confirmation"):
        return "answer" if state.get("needs_answer") else "confirmation_only"
    task_types = {t.type for t in state["tasks"]}
    return "validate" if task_types & {TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT} else "answer"


def _route_post_completeness(state: GraphState) -> str:
    return "answer" if state.get("needs_answer") else "analyze"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("query_catalog", query_catalog_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("delete_context", delete_context_node)
    graph.add_node("confirm_conflict", confirm_conflict_node)
    graph.add_node("decline_field", decline_field_node)
    graph.add_node("handle_split_intents", handle_split_intents_node)
    graph.add_node("validate_completeness", validate_completeness_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("complete_project", complete_project_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("analyze_context", analyze_context_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "CONTEXT_RETRIEVAL": "retrieve_context",
            "DIRECT_ANSWER": "generate_answer",
            "DATABASE_RETRIEVAL": "query_catalog",
            "CONTEXT_UPDATE": "build_context",
            "CONTEXT_DELETE": "delete_context",
            "decline_field": "decline_field",
            "confirm_conflict": "confirm_conflict",
            "operation_questions_pending": "analyze_context",
            "split": "handle_split_intents",
        },
    )
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("query_catalog", "generate_answer")
    graph.add_edge("generate_answer", "analyze_context")

    graph.add_conditional_edges(
        "decline_field",
        _route_decline,
        {"rephrasing": "analyze_context", "continue": "validate_completeness"},
    )
    graph.add_edge("confirm_conflict", "validate_completeness")

    graph.add_conditional_edges(
        "build_context",
        _route_after_build_context,
        {"confirmation_pending": "analyze_context", "validate": "validate_completeness"},
    )
    graph.add_conditional_edges(
        "delete_context",
        _route_after_delete_context,
        {"confirmation_pending": "analyze_context", "validate": "validate_completeness"},
    )
    graph.add_conditional_edges(
        "handle_split_intents",
        _route_after_split,
        {"validate": "validate_completeness", "answer": "generate_answer", "confirmation_only": "analyze_context"},
    )
    graph.add_conditional_edges(
        "validate_completeness",
        _route_completeness,
        {"complete": "complete_project", "incomplete": "generate_question"},
    )
    graph.add_conditional_edges(
        "complete_project",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": "analyze_context"},
    )
    graph.add_conditional_edges(
        "generate_question",
        _route_post_completeness,
        {"answer": "generate_answer", "analyze": "analyze_context"},
    )
    graph.add_edge("analyze_context", END)

    return graph.compile()


app_graph = build_graph()
```

### 3.2 Graph diagram

```
                  START
                   │
                   ▼
              classify_intent  ◄── entry point
                   │  _route_intent
      ┌────────────┼──────────────────────────────┐
      ▼            ▼          ▼                  ▼
 retrieve_context  query_catalog  build_context  delete_context  handle_split_intents
      │            │          │ _route_after_    │ _route_after_    │ _route_after_split
      │            │          │ build_context    │ delete_context   │
      │            │          ▼                  ▼                  ▼
      │            │      analyze_context ◄┐  analyze_context ◄┐  validate_completeness ──► generate_answer ─┬─► analyze
      │            │      (confirmation)   │  (confirmation)   │      (split with writes)                    │
      │            │          │validate     │   │validate      │                                            │
      │            │          ▼             │   ▼              └► analyze_context (confirmation_only)        │
      │            │   validate_completeness◄┘   │                                                          │
      │            │          │ _route_completeness          decline_field  ──► analyze_context (rephrasing) │
      │            │   ┌──────┴───────┐                     (continue) ──► validate_completeness             │
      │            │   ▼              ▼                                                               │
      │            │ complete_project  generate_question     ────────────────────────────────────────┘
      │            │   │ _route_post    │ _route_post          generate_answer ──► analyze_context
      │            │   │ completeness   │ completeness          (sell/buy/answer edges)
      │            │   ▼                ▼
      │            │ analyze_context ◄─┘  analyze_context ──► END (normal)
      │            │   (answer branch)    (analyze branch)
      ▼            ▼
    generate_answer
      │
      ▼
    analyze_context
      │
      ▼
      END
```

Wait — the two cleanest full paths are:

- **Read/answer path:** `classify_intent → retrieve_context|query_catalog|generate_answer → generate_answer → analyze_context → END`
- **Write path:** `classify_intent → build_context|delete_context → validate_completeness → complete_project|generate_question → analyze_context → END`

And the **special shorts**:

- `pending_question` was produced → lots of branches *skip* completeness and
  go straight to `analyze_context`, which no-ops if `question_generated`.
- `confirm_conflict` after a yes/no answer → `validate_completeness` (a retract
  or a critical edit can reopen a knowledge gap).
- `decline_field` rephrase → `analyze_context`; exhausted → `validate_completeness`.

### 3.3 Edge-by-edge explanation

| From | Router | Returns / Maps to | Meaning |
|---|---|---|---|
| `classify_intent` | `_route_intent` | see §3.1 | The **single most important decision point** — everything downstream hangs off this. Priority: pending confirmation > pending operation questions > decline detection > >1 task (split) > single task intent. |
| `retrieve_context` | — (static) | `generate_answer` | A context retrieval always feeds the answer generator. |
| `query_catalog` | — (static) | `generate_answer` | A catalog search always feeds the answer generator. |
| `generate_answer` | — (static) | `analyze_context` | After answering, still check whether the project needs another question. |
| `decline_field` | `_route_decline` | `rephrasing → analyze_context`; `continue → validate_completeness` | A rephrase already queued a question (stop). Budget exhausted → let completeness decide. |
| `confirm_conflict` | — (static) | `validate_completeness` | A confirmed/declined critical write may reopen a gap. |
| `build_context` | `_route_after_build_context` | `confirmation_pending → analyze_context`; `validate → validate_completeness` | A held-back conflict means nothing was applied → no completeness check. |
| `delete_context` | `_route_after_delete_context` | `confirmation_pending → analyze_context`; `validate → validate_completeness` | Same posture; `question_generated` covers no-match/ambiguous replies. Only a real retraction validates. |
| `handle_split_intents` | `_route_after_split` | `validate / answer / confirmation_only` | Batch with writes → re-check completeness; without writes → answer; held-back conflict → just analyze. |
| `validate_completeness` | `_route_completeness` | `complete → complete_project`; `incomplete → generate_question` | The intake "done?" gate. |
| `complete_project` | `_route_post_completeness` | `answer → generate_answer`; `analyze → analyze_context` | A split turn that also asked an info question still gets its answer. |
| `generate_question` | `_route_post_completeness` | same as above | After asking, finish the turn (answer only if a split-batch needs it). |
| `analyze_context` | — (static) | `END` | Terminal reporting node. |

---

## 4. The `GraphState` Contract

`GraphState` is a `TypedDict` (not a pydantic model) — the schema LangGraph
uses to merge node outputs into the running state. Fields with the
`Annotated[type, operator.add]` annotation (`trace`) get **accumulated**
(concatenated) across node updates; every other field is a **last-writer-wins**
overwrite, unless a node simply doesn't return it.

```python
class GraphState(TypedDict):
    session_id: str
    project_id: str
    message: str
    history: str
    active_room_id: Optional[str]
    field_attempts: dict[str, int]
    intent: list[str]              # telemetry only (classify_operations labels)
    tasks: list[TaskSpec]          # ← the REAL routing signal
    retrieved: str
    pending_question: Optional[str]
    pending_gap: Optional[dict]            # serialized KnowledgeGap
    pending_confirmation: Optional[dict]   # serialized context_builder.Conflict
    pending_operation_questions: Optional[dict]
    operation_answers: Optional[dict[str, str]]
    update_summary: Optional[str]
    answer: str
    complete: bool
    needs_answer: bool
    question_generated: bool
    wrapup_message: Optional[str]
    trace: Annotated[list[TraceEntry], operator.add]
```

Field responsibilities:

- **`tasks`** — the list of `TaskSpec`s produced by classification. **This,
  not `intent`, drives routing.**
- **`intent`** — legacy single-string labels, now just human-readable telemetry.
- **`retrieved`** — either `retrieve_context`'s graph summary or
  `query_catalog`'s catalog results, later fed to `generate_answer`.
- **`pending_*` family** — one-turn-at-a-time dialogue holds that get
  round-tripped into `ChatSession` so the *next* turn resumes them.
- **`question_generated`** — a latch flag: some node already produced this
  turn's trailing question/confirmation, so `analyze_context`/others must not
  generate a second one.
- **`operation_answers`** — this turn's structured reply `{op_id: value}` used
  only by `classify_intent_node`'s resume branch.
- **`complete`** — `validate_completeness_node`'s verdict (used by chat.py to
  set session status).
- **`trace`** — accumulates a `TraceEntry` per node (LLM capture included) for
  the Process Trace panel.

---

## 5. `classify_intent_node` in Detail

### 5.1 Purpose

`classify_intent_node` is the **entry point** of the graph. It is a thin
wrapper over `app/understanding.understand()`, which itself wraps
`llm.classify_operations()` — a single structured LLM call that splits the
message into independent **operations** and labels each with one of five
intents, while grounding each against the live project tree via a `connection`
field.

The node has two modes:

1. **Fresh classification** — no `pending_operation_questions` in state →
   call `understand()`.
2. **Resume** — `pending_operation_questions` is present → skip the LLM
   entirely, reload the stashed tasks, merge the user's structured answers
   (`operation_answers`) into their `connection` fields.

After either mode, an **unconditional gate** runs: if any
`EDIT_CONTEXT`/`DELETE_CONTEXT` task still has `connection=None`, the node
holds the whole turn and returns clarifying questions instead of letting the
write proceed.

### 5.2 Complete code (`app/graph.py:261-353`)

```python
async def classify_intent_node(state: GraphState) -> dict:
    start = time.perf_counter()
    with _node_span("classify_intent") as span:
        pending = state.get("pending_operation_questions")
        capture: dict = {}
        if pending:
            # ── RESUME: reply to last turn's clarifying questions ──
            answers = state.get("operation_answers") or {}
            tasks = [TaskSpec(**t) for t in pending["tasks"]]
            for task in tasks:
                if task.op_id in answers:
                    task.connection = answers[task.op_id]
            raw_intents = [understanding.TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
            output_summary = f"resumed {len(tasks)} operation(s), {len(answers)} answer(s) applied"
        else:
            # ── FRESH: classify this turn's message ──
            tree_text = await context_builder.render_project_tree_text(state["project_id"])
            meaning = await understanding.understand(
                state["message"],
                state["history"],
                pending_field=(state.get("pending_gap") or {}).get("canonical_path"),
                tree_text=tree_text,
                capture=capture,
            )
            tasks = meaning.tasks
            raw_intents = [understanding.TASK_TYPE_TO_INTENT.get(t.type, t.type.value.lower()) for t in tasks]
            output_summary = ",".join(raw_intents)
            if raw_intents != meaning.raw_intents:
                output_summary += f" (guard adjusted from: {','.join(meaning.raw_intents)})"

        entry = _trace(
            "classify_intent", settings.model_intent_classifier, state["message"],
            output_summary, start,
            llm_input=capture.get("messages"), llm_output=capture.get("raw_output"),
        )
        span.update(input=entry.input_summary, output=entry.output_summary)

    # ── UNCONDITIONAL GROUNDING GATE ──
    unresolved = _unresolved_connection_tasks(tasks)   # write tasks with connection = None
    if unresolved:
        with _node_span("generate_operation_questions") as cspan:
            questions, clarify_entry = await _generate_operation_questions(state["project_id"], unresolved)
            cspan.update(input=", ".join(t.target for t in unresolved),
                         output=f"{len(questions)} operation(s) held for clarification")
        return {
            "intent": raw_intents,
            "tasks": tasks,
            "pending_operation_questions": {"tasks": [t.model_dump() for t in tasks], "questions": questions},
            "question_generated": True,
            "trace": [entry, clarify_entry],
        }
    return {"intent": raw_intents, "tasks": tasks, "pending_operation_questions": None, "trace": [entry]}
```

### 5.3 What the node returns, and why that matters

- **Return 1 (clean):** `{"intent", "tasks", "pending_operation_questions": None, "trace"}`.
  Nothing is held → `_route_intent` proceeds to normal routing for the tasks.
- **Return 2 (held):** `pending_operation_questions` is populated with the
  **entire** task list (resolved and unresolved) plus one question per
  unresolved op. It also sets `question_generated=True`. `_route_intent` sees
  `pending_operation_questions` and returns `"operation_questions_pending"` →
  straight to `analyze_context` → END. Nothing executes this turn; the write is
  blocked until the user answers.

The resume trick: because the stashed payload holds **all** tasks, a
partially-answered resume re-asks only the still-unresolved ops — answered ones
keep their `connection` and would pass the gate next time.

### 5.4 `_generate_operation_questions` — one batched LLM call

(`app/graph.py:178-254`) The old code called the LLM once **per** unresolved op
via `asyncio.gather`. Now there is exactly **one** batched call,
`llm.resolve_room_connections()`, fed the rendered project tree plus each
task's text/intent. The Room Resolution Agent prompt (`app/prompts.py:431`)
always returns `(question, [options...])` for every operation in input order —
there is deliberately no "resolvable without asking" verdict, so the
always-block behavior is a property of the prompt, not Python-side confidence
branching.

```python
async def _generate_operation_questions(project_id, tasks) -> tuple[list[dict], TraceEntry]:
    tree_text = await context_builder.render_project_tree_text(project_id)
    operations = [
        {
            "text": task.target,
            "intent": understanding.TASK_TYPE_TO_INTENT.get(task.type, task.type.value.lower()),
            "connection": None,
        }
        for task in tasks
    ]
    capture: dict = {}
    call_start = time.perf_counter()
    results = await llm.resolve_room_connections(tree_text, operations, capture=capture)
    fallback_options = await _fallback_clarification_options(project_id)

    questions: list[dict] = []
    for i, task in enumerate(tasks):
        result = results[i] if i < len(results) else None
        if result is not None and result.question and result.options:
            question_text = result.question
            options = [o.model_dump() for o in result.options]
        else:
            question_text = f'Which room is "{task.target}" for?'
            options = fallback_options                    # defensive backstop
        questions.append({
            "op_id": task.op_id,
            "text": task.target,
            "question": question_text,
            "options": options,
            "allow_custom": True,                          # free-text allowed too
        })
    # ... builds one TraceEntry for the whole batched call ...
    return questions, entry
```

`_fallback_clarification_options` (`app/graph.py:158-175`) is a deterministic
safety net using the project's own live room list (capped at 3 named rooms +
a "Something else / a new room" catch-all) used only when the batched response
comes back short or malformed.

### 5.5 The grounding gate helper

```python
_UNRESOLVED_CONNECTION_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT)

def _unresolved_connection_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    return [t for t in tasks if t.type in _UNRESOLVED_CONNECTION_TYPES and t.connection is None]
```

This is checked on **every** turn, single-op included. A bare
"add a laminate" with no room in scope cannot be grounded by the classifier,
so the turn *waits* rather than the system guessing.

---

## 6. The Understanding Layer (what `classify_intent` actually calls)

### 6.1 `app/understanding.py` (complete, verbatim)

```python
"""Pure intent-understanding layer — understand() reads a message plus a
minimal serialized slice of context and returns a Meaning. It never touches
Mongo, never receives a session/state object, and has no side effects."""

from pydantic import BaseModel
from app import llm
from app.models import FIELD_LABELS
from app.tasks import TaskSpec, TaskType

# Mechanical relabeling of classify_operations' 5 intent strings onto TaskType.
_INTENT_TO_TASK_TYPE: dict[str, TaskType] = {
    "CONTEXT_UPDATE": TaskType.EDIT_CONTEXT,
    "CONTEXT_DELETE": TaskType.DELETE_CONTEXT,
    "CONTEXT_RETRIEVAL": TaskType.RETRIEVE_CONTEXT,
    "DATABASE_RETRIEVAL": TaskType.DATABASE_QUERY,
    "DIRECT_ANSWER": TaskType.ANSWER,
}
TASK_TYPE_TO_INTENT: dict[TaskType, str] = {v: k for k, v in _INTENT_TO_TASK_TYPE.items()}


class Meaning(BaseModel):
    tasks: list[TaskSpec]
    raw_message: str
    raw_intents: list[str]     # telemetry only


def _operations_to_tasks(operations: list["llm.Operation"]) -> list[TaskSpec]:
    return [
        TaskSpec(
            type=_INTENT_TO_TASK_TYPE[op.intent],
            op_id=op.id,
            target=op.text,
            connection=op.connection,
        )
        for op in operations
    ]


async def understand(message, history="", pending_field=None, *, tree_text="Project", capture=None) -> Meaning:
    operations = await llm.classify_operations(
        message, history, pending_field=_pending_field_label(pending_field), tree_text=tree_text, capture=capture
    )
    # RULE 15: unclear/no-op input → empty list → fall back to a single DIRECT_ANSWER
    if not operations:
        operations = [llm.Operation(id="op_1", text=message, intent="DIRECT_ANSWER")]
    tasks = _operations_to_tasks(operations)
    raw_intents = [op.intent for op in operations]
    return Meaning(tasks=tasks, raw_message=message, raw_intents=raw_intents)
```

`_pending_field_label` converts a raw `room:<id>.style` pending-field key to a
human label (`style`) for the classifier's prompt.

### 6.2 `app/llm.py` — the structured classifier

```python
OperationIntent = Literal[
    "CONTEXT_UPDATE", "CONTEXT_DELETE", "CONTEXT_RETRIEVAL", "DATABASE_RETRIEVAL", "DIRECT_ANSWER"
]

class Operation(BaseModel):
    text: str = Field(description="the original meaningful operation text ...")
    intent: OperationIntent
    connection: Optional[str] = None   # canonical path | new room name | Project | None
    id: str = ""                       # assigned after parsing, "op_1", "op_2", ...

class OperationClassification(BaseModel):
    operations: list[Operation] = Field(default_factory=list)
```

```python
async def classify_operations(message, history="", pending_field=None, *, tree_text="Project", capture=None) -> list[Operation]:
    operations = await _classify_operations_retrying(message, history, pending_field, tree_text, capture=capture)
    for i, op in enumerate(operations, start=1):
        op.id = f"op_{i}"
    return operations
```

The actual model call (`_classify_operations_raw`):

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
    extra_body=_REASONING_KWARGS,
)
```

Robustness notes:

- The prompt (`CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE`, `app/prompts.py`) contains
  the **rendered live project tree** (`tree_text` comes from
  `context_builder.render_project_tree_text`). That's what lets the classifier
  return a **grounded `connection`** (a real canonical path into the current
  graph) rather than a hallucinated one.
- `_salvage_operations` recovers an `OperationClassification` from the raw text
  of failed attempts when instructor's TOOLS-mode parsing hiccups.
- Retry ladder: 1 attempt, then up to 4 more with salvage between each.

### 6.3 The `TaskSpec` vocabulary (`app/tasks.py`)

```python
class TaskType(str, Enum):
    ANSWER = "ANSWER"
    EDIT_CONTEXT = "EDIT_CONTEXT"
    DELETE_CONTEXT = "DELETE_CONTEXT"
    RETRIEVE_CONTEXT = "RETRIEVE_CONTEXT"
    DATABASE_QUERY = "DATABASE_QUERY"

class TaskSpec(BaseModel):
    type: TaskType
    op_id: str                 # "op_1", "op_2" — key for operation_answers replies
    target: Optional[str]      # this operation's own text
    room_hint: Optional[str]   # raw room-type NAME, rarely set directly anymore
    connection: Optional[str]  # classifier's grounding: canonical path | new room | None
```

`connection` is the centerpiece of the current design:
- `None` → the classifier couldn't place the write → clarifying question.
- a path like `Rooms.<id>.Materials.countertop` → a *grounded* anchor.
- a free-text room name → a *hint* that gets fuzzy-matched downstream.

---

## 7. Every Downstream Node, Explained

### 7.1 `retrieve_context_node` — `CONTEXT_RETRIEVAL`

```python
async def retrieve_context_node(state: GraphState, task: Optional[TaskSpec] = None) -> dict:
    message = (task.target if task else None) or state["message"]
    nodes = await retrieval.retrieve_scoped(message, state["project_id"])
    parts = [f"{field_name}={node.value}" for node in nodes if node.value is not None]
    summary = ", ".join(parts) or "no context yet"
    return {"retrieved": summary, "trace": [entry]}
```

Pulls the **relevant subtree only** from the knowledge graph
(`app/retrieval.py` fuzzy-matches a room name in the query, then walks the
tree at depth ≤ 2), formats it as `field=value` pairs, and stores it in
`state["retrieved"]`. Static edge → `generate_answer`.

### 7.2 `query_catalog_node` — `DATABASE_RETRIEVAL`

```python
items = await rag.query_catalog(message)   # $vectorSearch on CatalogItem.embedding
summary = "; ".join(f"{i.title}: {i.description[:80]}" for i in items) or "no matches"
return {"retrieved": summary, "trace": [entry]}
```

Semantic search over the Mongo catalog. Also feeds `state["retrieved"]` and
static-edges to `generate_answer`.

### 7.3 `build_context_node` — `CONTEXT_UPDATE`

The primary **write** path. Replaced the old `extract_fields` +
`update_context_graph` pair with `context_builder.build_context()`.

```python
async def build_context_node(state, task=None, resolved=None) -> dict:
    message = (task.target if task else None) or state["message"]
    connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection if task else None)
    active_room_id = connection_room_id or state.get("active_room_id")
    room_hint = connection_room_hint or (task.room_hint if task else None)

    if resolved is not None:
        result = await context_builder.commit_context(resolved)   # reuse pre-resolved plan
    else:
        result = await build_context(message, state["project_id"], active_room_id, room_hint=room_hint)

    if result.pending_confirmations:                       # critical-tier value change
        conflict = result.pending_confirmations[0]
        question = await llm.generate_conflict_confirmation(field_label, conflict.old_value, conflict.new_value, ...)
        return {
            "active_room_id": result.room_id,
            "pending_confirmation": {**conflict.model_dump(), "field_label": ..., "question": question},
            "question_generated": True,
            "trace": [entry, confirm_entry],
        }

    update_summary = await _summarize_written(state["project_id"], result.written)
    return {"active_room_id": result.room_id, "update_summary": update_summary or None, "trace": [entry]}
```

Inside `context_builder.build_context` the pipeline is:
1. `extract_entities` (structured `ExtractedFields` extraction).
2. resolve the room (`_resolve_room`, fuzzy-match or allocate new).
3. route each structured field → a `ProposedWrite` at its canonical path;
   freeform mentions → `map_to_canonical` (embedding-based matching).
4. commit via `apply_to_graph`/`commit_context`, detecting **conflicts** —
   a critical-tier field that would *change* an existing value is held back as
   a `pending_confirmation` instead of silently overwritten.

Two exit routes:
- **Conflict held** → returns `pending_confirmation` + `question_generated=True`;
  router sends to `analyze_context` (skip completeness — nothing changed yet).
- **Written cleanly** → returns `update_summary` ("Got it — noted …"); router
  sends to `validate_completeness`.

### 7.4 `delete_context_node` — `CONTEXT_DELETE`

Handles removals, with a strict "never guess" posture:

```python
# 1. room-level deletion?  (target text fuzzy-matches a whole room name)
room_match = await _match_room_for_deletion(project_id, target_text)
if room_match is not None:
    return await _handle_room_deletion(state, room_match, start, span)   # holds for confirmation

# 2. item-level fallback among live freeform facts
match = await canonical_mapper.resolve_deletion_target(target_text, project_id, room_id)
if match.outcome == "none":        return {"pending_question": ..., "question_generated": True, ...}
if match.outcome == "ambiguous":   return {"pending_question": "did you mean: ...?", ...}
if field_tier == "critical":       return {"pending_confirmation": {...}, "question_generated": True, ...}
await versioning.retract_node(node)
return {"update_summary": f"Removed {label!r}.", "trace": [entry]}
```

`versioning.retract_node` sets `lifecycle="retracted"` and appends a
`KnowledgeNodeVersion` row — the source of truth for the deletion.

### 7.5 `confirm_conflict_node` — answer to a held-back conflict

```python
kind = conflict.get("kind", "edit")      # "edit" | "delete" | "delete_room"
if _is_affirmative(message):
    if kind in ("delete", "delete_room"):
        # versioning.retract_node / versioning.retract_subtree
    else:
        await apply_to_graph(project_id, [ProposedWrite(path, type, new_value, tier)])
elif _is_negative(message):
    outcome = f"declined: kept {conflict['old_value']!r}"
else:  # ambiguous — never apply an unconfirmed change
    outcome = f"ambiguous reply — kept {conflict['old_value']!r}"
return {"pending_confirmation": None, "trace": [entry]}
```

`_is_affirmative`/`_is_negative` are narrow deterministic phrase lists
(`app/graph.py:118-131`) — no extra LLM call for a short yes/no.

### 7.6 `decline_field_node` — user said "skip/I don't know"

```python
attempts = state["field_attempts"].get(canonical_path, 0) + 1
limit = RETRY_LIMITS.get(tier, 0)         # critical:3, moderate:1, optional:0
if attempts <= limit:
    question = await generate_question(KnowledgeGap(**gap_dict), context, is_retry=True)  # rephrase
    return {"field_attempts": ..., "pending_question": question, "pending_gap": gap_dict, "question_generated": True, ...}
context = await context_builder.known_fields(project_id, room_id)
await inference.infer_field(...)          # budget exhausted → infer a reasonable value
return {"field_attempts": ..., "pending_question": None, "pending_gap": None, ...}
```

So a field is repfhrased within its retry budget, then **inferred** (marked
`changed_by="inferred"`) so the project can still complete.

### 7.7 `handle_split_intents_node` — more than one operation

```python
async def handle_split_intents_node(state: GraphState) -> dict:
    result = await execution.execute(state["tasks"], state)   # see §8
    result["trace"] = [entry, *result.get("trace", [])]
    return result
```

It's a thin delegate to `app/execution.py` — **not** LangGraph fan-out (see §8).

### 7.8 `validate_completeness_node`

```python
gap = await find_knowledge_gap(state["project_id"], state.get("active_room_id"))
complete = gap is None
return {"complete": complete, "trace": [entry]}
```

`find_knowledge_gap` (`app/question_engine.py`) walks `ontology/v1.yaml` paths
(ProjectType → Budget → rooms' fields in priority order → Timeline) and returns
the first missing field as a `KnowledgeGap`.

### 7.9 `generate_question_node`

```python
gap = await find_knowledge_gap(...)                     # guaranteed to exist here
context = await context_builder.known_fields(...)
question = await generate_question(gap, context)        # llm.generate_question
return {"pending_question": question, "pending_gap": gap.model_dump(), "question_generated": True, ...}
```

Queues the next intake question; `pending_gap` lets the *next* turn attribute a
decline to this exact field without re-deriving it.

### 7.10 `complete_project_node`

```python
summary = await context_builder.materialize_project_summary(project_id)   # flat snapshot off the graph
assumptions = await _build_assumptions(project_id)                        # provenance: inferred/calculated
project = ProjectContext(...); await project.insert()                     # Mongo snapshot doc
wrapup_message = await llm.generate_wrapup_message(summary)               # closing line
return {"wrapup_message": wrapup_message, "question_generated": True, ...}
```

No inference sweep here — `decline_field_node` already inferred anything that
wasn't answered. Materials `ProjectContext` snapshot + closing wrap-up.

### 7.11 `generate_answer_node`

Runs two tasks **concurrently** via `asyncio.gather`:
- `_answer()` streams tokens from `llm.generate_answer(...)` and emits them as
  custom stream events (`answer_token`).
- `_question()` checks `question_generated` — if some other node already queued
  this turn's trailing question/confirmation, it skips; otherwise it looks up a
  knowledge gap and, if one exists, generates the question.

```python
(answer, answer_entry), (question, gap_dict, question_entry) = await asyncio.gather(_answer(), _question())
result = {"answer": answer, "trace": [answer_entry]}
if question_entry is not None:
    result["pending_question"] = question
    result["pending_gap"] = gap_dict
    result["question_generated"] = True
    result["trace"].append(question_entry)
```

This is how a read turn can both answer AND ask the next intake question in one
round trip.

### 7.12 `analyze_context_node`

Runs after **every** branch as the terminal node:

```python
if state.get("question_generated"):   # something already asked this turn → no-op
    return {"trace": [entry]}
gap = await find_knowledge_gap(...)
if gap is None:
    return {"trace": [entry]}          # project complete → no question
context = await context_builder.known_fields(...)
question = await generate_question(gap, context)
return {"pending_question": question, "pending_gap": gap.model_dump(), "question_generated": True, ...}
```

It guarantees an unrelated question mid-flow doesn't get dropped — e.g. an
update turn that didn't itself queue anything still asks the next gap question.

---

## 8. Multi-Operation Batch Execution (`app/execution.py`)

When `_route_intent` returns `"split"`, `handle_split_intents_node` delegates to
`execution.execute(tasks, state)`. This mirrors LangGraph fan-out but runs in
plain asyncio:

```python
_read_dispatch = {TaskType.DATABASE_QUERY: query_catalog_node, TaskType.RETRIEVE_CONTEXT: retrieve_context_node}

async def execute(tasks, state) -> dict:
    read_dispatch = {...}
    write_tasks = [t for t in tasks if t.type in _CONTEXT_TASK_TYPES]       # EDIT/DELETE_CONTEXT
    read_tasks = [t for t in tasks if t.type in read_dispatch]
    writer = _stream_writer()

    # Reads start immediately, in parallel, never wait on writes.
    reads_future = asyncio.gather(*(_dispatch_read_task(t, state, read_dispatch, writer) for t in read_tasks))

    if write_tasks:
        # 1) PARALLEL RESOLVE — what would each task touch? (no writes yet)
        resolved_tasks = await asyncio.gather(*(_resolve_write_task(t, state) for t in write_tasks))
        # 2) CLUSTER tasks that touch overlapping targets (union-find over key sets)
        clusters = _cluster(resolved_tasks)
        # 3) COMMIT each cluster sequentially; every cluster in parallel
        cluster_results = await asyncio.gather(*(_commit_cluster(cluster, state, writer) for cluster in clusters))
        write_results = [item for cluster in cluster_results for item in cluster]
    else:
        write_results = []

    raw_read_results = await reads_future
    # merge everything into one partial-state dict (trace, update_summary,
    # retrieved, active_room_id, first pending_confirmation/question, needs_answer)
    return merged
```

Key design rules:

1. **Reads never wait for writes** — they all start immediately.
2. **Resolve-then-cluster-commit**: every write task is *resolved* (read-only)
   in parallel first, then tasks whose target keys overlap (same canonical
   path, same room, or same **not-yet-created** room name) are grouped.
3. `_commit_cluster` commits **deletes before edits**, sequentially — the first
   task in a cluster reuses its pre-resolved plan; the rest re-resolve fresh so
   they observe earlier same-cluster writes. This is how two ops that both want
   to create the same brand-new room create it *once*.
4. One failure doesn't kill a batch — every task's outcome is captured
   (success dict or `Exception`), and failures become `TraceEntry`s.
5. `_dispatch_read_task`/`_commit_cluster` emit an `operation_progress` custom
   stream event per task for incremental SSE progress.

---

## 9. State Persistence: Mongo (sessions) + Neo4j (knowledge graph)

### 9.1 `ChatSession` (Mongo, `app/models.py:117`)

Holds *only dialogue mechanics* — project facts live in Neo4j:

```python
class ChatSession(Document):
    session_id: str
    project_id: str                      # minted at session creation
    messages: list[Message]
    trace: list[TraceEntry]
    status: Literal["in_progress", "complete"]
    active_room_id: Optional[str]
    field_attempts: dict[str, int]       # decline retry counters
    pending_gap: Optional[dict]          # serialized KnowledgeGap
    pending_confirmation: Optional[dict] # serialized context_builder.Conflict
    pending_operation_questions: Optional[dict]
    created_at / updated_at
```

Every turn these are loaded (`get_or_create_session`), injected into
`GraphState`, and written back (`save_session`).

### 9.2 `KnowledgeNode` (+ versions) — Neo4j

```python
class KnowledgeNode(BaseModel):
    node_id: str
    canonical_path: str          # e.g. "Project.Rooms.r1.Style"
    parent_id: Optional[str]     # derived from :CHILD_OF edge, not stored
    node_type: str               # matches ontology/v1.yaml, e.g. "Style"
    value: Optional[Any]
    status: FieldStatus          # "confirmed" | "assumed"
    lifecycle: Literal["active", "retracted"]   # deletions
    changed_by: ChangedBy        # "user_message" | "inferred" | "system_default"
    project_id: str
    room_id: Optional[str]
    embedding: Optional[list[float]]
```

Written by `context_builder.apply_to_graph` / `canonical_mapper`; deletion sets
`lifecycle="retracted"` via `versioning.retract_node` (append-only history in
`KnowledgeNodeVersion`). The knowledge graph is the live source of truth; the
`ProjectContext` Mongo doc at completion is just an archival snapshot.

### 9.3 The round-trip loop (why answers come back in a *later* graph run)

Because LangGraph state doesn't persist across HTTP requests, everything the
next turn needs to know is written into `ChatSession`:

```
Turn N  build_context/classify_intent sets  pending_confirmation /
                                            pending_operation_questions / pending_question + pending_gap
        run_chat_turn persists them on ChatSession
Turn N+1 run_chat_turn loads them into GraphState
        classify_intent_node / confirm_conflict_node graciously resumes
```

`operation_answers` is the *structured* reply channel for `operation_questions`
(the only reply to prior questions that isn't plain message text).

---

## 10. Complete End-to-End Walkthrough Scenarios

### Scenario A — plain question ("what's my kitchen budget?")

1. `POST /chat` → `run_chat_turn` seeds state.
2. `classify_intent_node` → `understand()` → `classify_operations` →
   `[Operation(text="what's my kitchen budget", intent="CONTEXT_RETRIEVAL", connection=...)]`
   → `[TaskSpec(type=RETRIEVE_CONTEXT)]`. No write task → gate passes.
3. `_route_intent` → `RETRIEVE_CONTEXT` → `retrieve_context` → `retrieve_scoped`
   pulls the kitchen subtree → `state["retrieved"]`.
4. static edge → `generate_answer` → streams answer tokens; the concurrent
   `_question` looks up a knowledge gap (maybe none → no question).
5. `analyze_context` → END. Session saved. SSE events: progress/retrieve,
   progress/generate_answer, token…, done.

### Scenario B — grounded write ("make the kitchen modern style, 200 sqft")

1. `classify_operations` → `[Operation(intent="CONTEXT_UPDATE", text=...,
   connection="Rooms.<id>.Style")]` → `TaskSpec(EDIT_CONTEXT)`.
2. Gate passes (`connection` set) → route → `build_context`.
3. `build_context` resolves room, extracts fields, proposes writes, commits.
   Style is a moderate-tier field → applies directly, no conflict.
4. `_route_after_build_context` → `validate` → `validate_completeness` →
   maybe a gap exists → `generate_question` → `pending_question` →
   `analyze_context` → END. SSE: context_updated + ask_question.

### Scenario C — ungrounded write (the "block" behavior)

1. User: "add a laminate" (no room). `classify_operations` returns
   `connection=None` for the EDIT_CONTEXT op.
2. `classify_intent_node` finds `unresolved` → calls
   `resolve_room_connections` (ONE batched LLM call) → returns a question +
   options for the op.
3. Node returns `pending_operation_questions` + `question_generated=True`;
   `_route_intent` → `operation_questions_pending` → `analyze_context` → END.
   **Nothing is written.** SSE: `operation_questions` event.
4. User picks an option → next `POST /chat` sends `operation_answers={"op_1": "...label..."}`.
5. `classify_intent_node` **resume branch**: no LLM, tasks reloaded,
   `op_1.connection` set → gate passes → routing proceeds exactly as Scenario B.

### Scenario D — critical change held for confirmation

1. User: "actually make the total budget 20 lakh" and `Project.Budget.Total`
   already exists (critical tier).
2. `build_context` extracts `overallBudget="20 lakh"` → `detect_conflicts` sees
   a critical value *change* → `build_context_node` returns `pending_confirmation`.
3. Router → `analyze_context` (skips completeness). SSE: `confirm_change`.
4. Next turn `confirm_conflict_node`: "yes" → `apply_to_graph` new value →
   `validate_completeness`; "no"/ambiguous → keep old value.

### Scenario E — mixed/split turn ("remove the rug and what's the budget?")

1. `classify_operations` → `[DELETE_CONTEXT, CONTEXT_RETRIEVAL]` → 2 tasks →
   `_route_intent` → `split` → `handle_split_intents` → `execution.execute`.
2. Reads start immediately (`retrieve`); the delete resolves/clusters/commits.
3. `_route_after_split` → `task_types & {EDIT,DELETE}` → `validate` because a
   deletion can reopen a gap; then `_route_post_completeness` →
   `needs_answer=True` → `generate_answer` → merge → `analyze_context`.

---

## 11. Observability, SSE Events & the Viewer

### 11.1 SSE events produced by `run_chat_turn`

| Event | Meaning |
|---|---|
| `image_description` | vision description prepended to the message |
| `token` | streamed answer token from `generate_answer_node` (custom mode) |
| `operation_progress` | per-task progress from `execution.execute` (custom mode) |
| `progress` | a graph node started (updates mode) |
| `trace` | node's `TraceEntry` list for the Process Trace panel |
| `context_updated` | `update_summary` after a clean write |
| `confirm_change` | critical-tier change held for yes/no |
| `operation_questions` | clarifying room-selection questions (+ options) |
| `ask_question` | next knowledge-gap intake question |
| `wrapup` | closing line when the project reached complete |
| `error` | graph failure |
| `done` | turn finished (session_id, status) |
| `heartbeat` | keep-alive comment line |

`stream_mode=["custom", "updates", "values"]` supplies all three flavors:
custom (tokens/operation_progress), updates (per-node trace), values (final
state).

### 11.2 Langfuse + trace

Every node in `app/graph.py` opens a `_node_span` under the turn's root
`chat_turn` span; each real LLM call nests under it with token usage/cost.
`_trace(...)` builds a `TraceEntry` with full `llm_input`/`llm_output` (for the
viewer's Process Trace panel), while the span keeps the granular waterfall.

---

## 12. Key File Map & Line References

| Concern | File | Lines |
|---|---|---|
| Graph wiring / all nodes / routers | `app/graph.py` | `build_graph` 1136–1220; `app_graph` 1223 |
| → the controller after classify | `app/graph.py` | `_route_intent` 1056–1080 |
| → entry node | `app/graph.py` | `classify_intent_node` 261–353 |
| → batched clarification | `app/graph.py` | `_generate_operation_questions` 178–254 |
| GraphState schema | `app/graph.py` | 31–76 |
| Understanding (pure wrapper) | `app/understanding.py` | full (103 lines) |
| Classifier + all LLM calls | `app/llm.py` | `classify_operations` 162–227; `resolve_room_connections` 297–347 |
| Task vocabulary | `app/tasks.py` | full (66 lines) |
| Turn outer loop / SSE | `app/chat.py` | `run_chat_turn` 44–273; `/chat` 276–286 |
| Multi-op dispatcher | `app/execution.py` | `execute` 238–324 |
| Write pipeline | `app/context_builder.py` | `build_context`, `commit_context`, `render_project_tree_text` 316+ |
| Knowledge-gap engine | `app/question_engine.py` | `find_knowledge_gap` 68–117 |
| Scoped graph retrieval | `app/retrieval.py` | full (70 lines) |
| Catalog semantic search | `app/rag.py` | full (26 lines) |
| Session + node models | `app/models.py` | `ChatSession` 117–152; `KnowledgeNode` 185–254 |
| Deletion/versioning | `app/versioning.py`, `app/canonical_mapper.py` | — |
| Model selection / keys | `app/config.py` | 42–88 |
| FastAPI app | `app/main.py` | full (42 lines) |
| Prompts (classify/room-resolution/etc.) | `app/prompts.py` | `classify_operations_system` 398; Room Resolution Agent 431+ |

---

## TL;DR — the answer to your question

**Yes, it is `build_graph()` — but concretely it's the compiled LangGraph
`StateGraph` (`app_graph = build_graph()` at `app/graph.py:1223`) running on
top of LangGraph's runtime.** `classify_intent_node` never picks the next node
itself; after it returns, LangGraph automatically invokes the conditional-edge
router **`_route_intent(state)`** (`app/graph.py:1056`), which inspects
`pending_confirmation` / `pending_operation_questions` / `state["tasks"]` /
`state["message"]` and returns the name of the next node. That returns-name →
run-node → returns-name cycle is LangGraph's runtime mechanism, and the full
set of possible nodes + edges that cycle may take is defined **only** in
`build_graph()`. Every other junction (`after build_context`, `after delete`,
`after split`, completeness, post-completeness, decline) works the same way,
each with its own small router function.