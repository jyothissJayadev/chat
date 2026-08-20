# Turn Flow & `generate_turn_reply` — How the Current System Actually Talks

This file has one job: explain, end to end, **what happens when a chat turn runs today** and
**exactly how `generate_turn_reply` turns raw pipeline output into the one message the user
sees**. It written from the current code (`app/pipeline.py`, `app/llm.py`, `app/prompts.py`,
`app/chat.py`, `app/graph.py`) after the turn-reply merge — the old two-call split
(`generate_turn_summary` for tables/next-question + `generate_answer` streamed separately for
direct questions) is gone. There is now **exactly one LLM call, exactly once per turn, that
produces every word the user reads.**

For the full per-stage system map see [SYSTEM_FLOW_OVERVIEW.md](SYSTEM_FLOW_OVERVIEW.md); for
per-function signatures see [MODULE_DETAILED_FLOW.md](MODULE_DETAILED_FLOW.md). This file is
narrower and deeper: one call, fully unpacked.

---

## 1. The one-sentence version

Every turn, after whatever writes/retrievals/searches happened, the pipeline builds a single
`pieces` dict and hands it to **one** LLM call — `llm.generate_turn_reply(pieces)` — which
returns a `TurnReply {reply, changes_summary, context_summary, is_question}`. `reply` is the
only thing that reads like a person talking; `changes_summary`/`context_summary` are markdown
tables for the UI, not conversation. Nothing else produces user-facing text.

---

## 2. Complete turn flow (where `generate_turn_reply` sits)

```
POST /chat
  │
  ▼
app/chat.py run_chat_turn()
  │  loads/creates ChatSession, optionally describes an attached image
  ▼
app/graph.py — LangGraph, 2 nodes
  │
  ├─ NODE 1: classify_intent_node
  │    • decline detection (skip/not-sure phrases)
  │    • resume branch (pending_operation_questions from a prior turn) OR
  │      fresh classification: llm.classify_operations() → TaskSpec list,
  │      each grounded to a graph `connection` (which room/entity it targets)
  │    • if any write/retrieve task can't be grounded → hold the turn,
  │      ask ONE clarifying question (llm.resolve_room_connections), END
  │
  ▼ (only if every task is grounded)
  NODE 2: run_pipeline_node → app/pipeline.py run_pipeline()
  │
  │  Partitions this turn's TaskSpecs into write / retrieval / query / answer.
  │
  │  Concurrent from the very start (don't need this turn's own writes):
  │    • DATABASE_QUERY tasks  → keywords → rag.query_catalog() (Atlas vector search)
  │    • project_context       → context_builder.known_fields(project_id, active_room_id)
  │                               (pre-write active room — just grounding, not this turn's deltas)
  │
  │  Sequential first (must land before anything reads the graph again):
  │    • write tasks (EDIT/DELETE) → _run_first_action()
  │        one combined llm.resolve_context_changes() call, then per-op apply
  │        to Neo4j → FirstActionResult{written, retracted, changes, active_room_id}
  │
  │  Then, now that writes have landed (must see POST-write state):
  │    • RETRIEVE_CONTEXT tasks → retrieval.load_subtree() per task's root
  │    • pending_gap            → question_engine.find_knowledge_gaps()
  │                               (KnowledgeGapBatch, or None = project complete)
  │
  │  ANSWER tasks never trigger their own LLM call any more — their `.target`
  │  text is simply joined into `direct_answer_request`, one field in `pieces`.
  │
  │  Once ALL of the above have resolved, build:
  │
  │    pieces = {
  │      "direct_answer_request": str | None,   # what the user actually asked, if anything
  │      "project_context":       dict,          # known fields, for grounding the answer
  │      "changes":               list[dict],    # every write this turn (before/after/action)
  │      "context_retrieval":     list[dict],    # every {field, value} this turn retrieved
  │      "database_results":      list[dict],    # every {title, description} catalog hit
  │      "pending_gap":           list[str]|None,# field label(s) still open, or None if complete
  │    }
  │
  │  ═══════════════════════════════════════════════════════════════════════
  │  ▶▶▶  llm.generate_turn_reply(pieces)  ◀◀◀   — THE SINGLE JOIN CALL
  │  ═══════════════════════════════════════════════════════════════════════
  │    Runs UNCONDITIONALLY. Every turn, no exceptions — even a turn where
  │    nothing happened still gets one call, so there is always a coherent
  │    reply to show. See §3-§5 below for exactly what happens inside it.
  │
  │  → TurnReply{reply, changes_summary, context_summary, is_question}
  │
  ▼
Back in app/chat.py:
  • appends the user message, then the assistant `reply`, to session.messages
  • session.status = "complete" if gaps==None else "in_progress"
  • stores pending_gap (ordinary) or pending_operation_questions (held turn)
  • save_session() → Mongo
  • SSE: pipeline_result{context_summary, changes_summary, reply, is_question} → done{session_id, status}
```

Two ordering rules worth internalizing, because they explain *why* the pipeline is shaped this
way:

1. **Writes must run before context-retrieval and gap-detection**, because both of those read
   the live graph — if they ran concurrently with the write, they could read pre-write state.
2. **`generate_turn_reply` must run dead last**, because it's the only step that needs
   *everything* — including this turn's own `changes` and the freshly-computed `pending_gap` —
   at once. There's no way to start it early.

Database query and `project_context` don't have either constraint (they don't depend on this
turn's writes), so they're kicked off immediately, in parallel with the write step, to shave
latency off the turn.

---

## 3. What used to happen (for contrast)

Before the merge, a turn with a direct question (e.g. "what is MDF?") produced text through
**two separate paths**:

- `_run_direct_answer()` fired `llm.generate_answer()` **concurrently** with everything else,
  streamed token-by-token as SSE `answer_token` → `token` events, and was appended to
  `session.messages` as its own assistant message.
- Everything else (changes made, context retrieved, database results, the next question) went
  through a **second**, separate call, `llm.generate_turn_summary()`, only fired if `need_summary`
  was true (`write_tasks or retrieval_tasks or query_tasks or gap_batch is not None`).

That meant a turn could produce **two assistant messages** with no connection between them — the
direct answer had no idea a catalog search happened in the same turn, and vice versa. It also
meant a pure `DIRECT_ANSWER`-only turn (no writes, no retrieval, no query, no gap) skipped
`generate_turn_summary` entirely (`need_summary=False`), so nothing ever mentioned there was
nothing left to ask.

## 4. What happens now — the merge

`generate_turn_reply` **always** runs, and it is handed the direct-answer text as just one more
input (`direct_answer_request`) alongside the rest. The model is instructed to weave everything
that applies — the answer, the catalog results, the changes, the next question — into **one**
flowing reply. There is no more streaming; the whole `reply` comes back as one structured field
and is shown to the user only once `pipeline_result` fires.

This is a genuine behavior change, not just a refactor: a turn that both asks "what is MDF?" and
triggers a laminate search now gets one answer that connects the two ("MDF is... — and here's a
laminate option that'd work well for that"), rather than a raw MDF definition followed
disconnectedly by a search-results paragraph.

---

## 5. Inside `generate_turn_reply` — the actual mechanics

`app/llm.py::generate_turn_reply(pieces, *, capture=None) -> TurnReply`

```python
messages = [
    {"role": "system", "content": prompts.build_turn_reply_system(pieces)},
    {"role": "user",   "content": prompts.turn_reply_user(pieces)},
]
```

Model: `settings.model_question_gen`, `max_tokens=1536`, `max_retries=2`, structured output via
`instructor` into the `TurnReply` Pydantic model.

### 5.1 The output shape

```python
class TurnReply(BaseModel):
    reply: str                        # the ONE conversational message
    changes_summary: Optional[str]    # markdown table, or None if `changes` was empty
    context_summary: Optional[str]    # markdown table, or None if `context_retrieval` was empty
    is_question: bool                 # True => reply ends on a question; False => closing line
```

Only `reply` is meant to be read as prose. `changes_summary` / `context_summary` are markdown
tables the frontend renders separately (see `app/static/viewer.html`) — the model is explicitly
told never to restate table values inside `reply`.

### 5.2 The system prompt is assembled per-turn, not static

This is the part most worth understanding: `TURN_SUMMARY_SYSTEM` used to be one fixed string.
Now `prompts.build_turn_reply_system(pieces)` **conditionally concatenates blocks**, so a turn
only ever sees instructions relevant to what actually happened this turn:

```python
def build_turn_reply_system(pieces: dict) -> str:
    parts = [TURN_REPLY_PREAMBLE.format(project_context=pieces.get("project_context") or {})]
    if pieces.get("direct_answer_request"):
        parts.append(ANSWER_BLOCK.format(direct_answer_request=pieces["direct_answer_request"]))
    if pieces.get("database_results"):
        parts.append(DATABASE_BLOCK.format(database_results=pieces["database_results"]))
    if pieces.get("changes"):
        parts.append(CHANGES_BLOCK)
    if pieces.get("context_retrieval"):
        parts.append(CONTEXT_BLOCK)
    parts.append(NEXT_STEP_BLOCK)
    parts.append(GENERAL_RULES_BLOCK)
    return "\n\n".join(parts)
```

| Block | Always included? | What it tells the model |
|---|---|---|
| `TURN_REPLY_PREAMBLE` | always | Sets the persona ("senior interior designer... warm, direct, conversational... never a form"), and injects `project_context` as grounding. |
| `ANSWER_BLOCK` | only if `direct_answer_request` | The client's actual question/message this turn. Instructs: answer small talk naturally and briefly; answer general design-knowledge questions directly and practically; answer project-specific questions using `PROJECT_CONTEXT`; be concise and opinionated; weave in database results if present rather than listing them separately; never invent project specifics not in `PROJECT_CONTEXT`. |
| `DATABASE_BLOCK` | only if `database_results` | The catalog hits found this turn. Instructs: fold as natural prose, never a table/bullet list; connect to the ANSWER section if one exists in the same reply; otherwise introduce the results standalone. |
| `CHANGES_BLOCK` | only if `changes` | Full spec for the `changes_summary` markdown table — one row per entry, columns `Item / Previous Value / New Value / Status`, exact status-label mapping (`created→"Added"`, `updated→"Updated"`, `deleted→"Removed"`, `deleted_room→"Room removed"`, `failed→"Could not apply ({reason})"`), and a hard rule to cover every entry, never merge/omit rows. |
| `CONTEXT_BLOCK` | only if `context_retrieval` | Spec for the `context_summary` table — columns `Field / Value`, values verbatim, no paraphrasing. |
| `NEXT_STEP_BLOCK` | always | Tells the model how to close `reply`: if `pending_gap` is non-empty, end on ONE natural question and set `is_question=true`; if empty, end on a short closing statement and set `is_question=false`; if literally nothing applied this turn, say so briefly and set `is_question=false`. |
| `GENERAL_RULES_BLOCK` | always | Cross-cutting rules: `reply` is the only conversational field, one flowing response, never section-labeled; tables live only in `changes_summary`/`context_summary`, never in `reply`; never repeat table values inside `reply`; never invent anything not grounded in the provided pieces or general design knowledge; empty list ⇒ table field is `None`, never an empty table. |

The practical effect: a turn that's a pure direct question (no writes, no retrieval, no catalog
search) sends the model a system prompt with just `PREAMBLE + ANSWER_BLOCK + NEXT_STEP_BLOCK +
GENERAL_RULES_BLOCK` — it never even sees the CHANGES/CONTEXT table-formatting instructions,
because there's nothing for it to produce a table from that turn.

### 5.3 The user message — the raw data itself

```python
def turn_reply_user(pieces: dict) -> str:
    return (
        "This turn's pieces (JSON):\n"
        f"{json.dumps(pieces, indent=2, default=str)}\n\n"
        "Cover every entry in `changes` and every entry in `context_retrieval` as its own "
        "table row — do not omit, merge, or summarize any of them away, however many there are."
    )
```

The entire `pieces` dict is serialized as JSON and handed over verbatim, with one explicit
completeness reminder repeated here (belt-and-braces alongside the CHANGES/CONTEXT block
instructions in the system prompt) — dropping a row silently is the failure mode this guards
against, since an LLM summarizing a long `changes` list is the most likely place to quietly
truncate.

### 5.4 Where each `pieces` field actually comes from

| Field | Producer | Notes |
|---|---|---|
| `direct_answer_request` | `" ".join(t.target for t in answer_tasks if t.target) or None` in `run_pipeline` | Every `ANSWER`-intent task's raw text for this turn, space-joined; `None` if there were no ANSWER tasks. |
| `project_context` | `context_builder.known_fields(project_id, state.get("active_room_id"))` | Fetched concurrently, using the turn's **pre-write** active room — it's grounding only, not meant to reflect this turn's own edits (those are already in `changes`). |
| `changes` | `first_action.changes if first_action else []` | From `_run_first_action`; each entry is `{path, before, after, action}`, `action ∈ created/updated/deleted/deleted_room/failed`. |
| `context_retrieval` | `_format_retrieved_nodes(retrieval_nodes)` | `[{field, value}, ...]`, built from whatever `RETRIEVE_CONTEXT` tasks pulled via `retrieval.load_subtree()`. |
| `database_results` | `query_results` | `[{title, description}, ...]` from `rag.query_catalog()`. |
| `pending_gap` | `[g.field_label for g in gap_batch.gaps] if gap_batch else None` | From `question_engine.find_knowledge_gaps()` — `None` specifically means the **whole project** is complete, not just this room. |

### 5.5 Trace / observability

Each call is wrapped in a `TraceEntry` (`node_name="generate_turn_reply"`,
`model_used=settings.model_question_gen`) recording the truncated input (`str(pieces)[:200]`),
the resulting `reply`, duration, and — if Langfuse capture is on — the full `messages` list sent
and the raw completion text. This is what powers the `trace` SSE event and any Langfuse
generation span for this step.

---

## 6. Worked example

User: *"The project is a 3BHK renovation, budget is 25 lakh. Add a modern kitchen around 300
sqft — show me laminate options. What is MDF?"*

1. Classification splits this into 4 ops: `CONTEXT_UPDATE` (project fields), `CONTEXT_UPDATE`
   (new room "Kitchen"), `DATABASE_RETRIEVAL` (laminates), `DIRECT_ANSWER` (MDF). All ground
   cleanly — no clarification hold.
2. Concurrently: database query runs (`generate_search_keywords` → `query_catalog`), and
   `project_context` is fetched. Meanwhile the first action writes `ProjectType`, `Budget.Total`,
   mints a room id for "Kitchen", writes `RoomType`/`Style`/`SquareFootage`.
3. After the write lands: gap detection finds the kitchen still needs `Budget` and `Timeline`
   (say) → `pending_gap = ["Kitchen budget", "Timeline"]`.
4. `pieces` ends up:
   ```json
   {
     "direct_answer_request": "What is MDF?",
     "project_context": {"projectType": "renovation", "overallBudget": "25 lakh", ...},
     "changes": [
       {"path": "Project.BasicInformation.ProjectType", "before": null, "after": "renovation", "action": "created"},
       {"path": "Project.Budget.Total", "before": null, "after": "25 lakh", "action": "created"},
       {"path": "Project.Rooms.<id>.RoomType", "before": null, "after": "Kitchen", "action": "created"},
       {"path": "Project.Rooms.<id>.Style", "before": null, "after": "modern", "action": "created"},
       {"path": "Project.Rooms.<id>.SquareFootage", "before": null, "after": "300", "action": "created"}
     ],
     "context_retrieval": [],
     "database_results": [{"title": "Sunmica Classic Laminate", "description": "..."}, ...],
     "pending_gap": ["Kitchen budget", "Timeline"]
   }
   ```
5. The system prompt this turn includes `PREAMBLE + ANSWER_BLOCK + DATABASE_BLOCK +
   CHANGES_BLOCK + NEXT_STEP_BLOCK + GENERAL_RULES_BLOCK` (no `CONTEXT_BLOCK`, since
   `context_retrieval` is empty).
6. `TurnReply` comes back roughly:
   - `reply`: *"MDF (medium-density fibreboard) is an engineered wood panel — smooth, stable,
     and a common laminate substrate, which is exactly what you'd want for the kitchen. I've
     noted the renovation, ₹25L budget, and a modern 300 sqft kitchen, and pulled a few laminate
     options like the Sunmica Classic that'd suit that style. What budget are you thinking for
     the kitchen specifically, and do you have a timeline in mind?"* — `is_question=true`
   - `changes_summary`: a 5-row markdown table, one row per entry above.
   - `context_summary`: `None` (nothing was retrieved this turn).
7. `app/chat.py` appends `reply` as the one assistant message, stores `pending_gap`, saves the
   session, and streams `pipeline_result` then `done{status: "in_progress"}`.

Compare this to the pre-merge version, which would have streamed a standalone MDF answer as
`token` events *and* separately emitted a summary with the table + next question — two
disconnected assistant messages instead of one.

---

## 7. Quick reference — call sites

| Function | File | Called by |
|---|---|---|
| `run_pipeline()` | `app/pipeline.py` | `app/graph.py::run_pipeline_node` (the graph's second and last node) |
| `generate_turn_reply()` | `app/llm.py` | `run_pipeline()`, once, unconditionally, last |
| `build_turn_reply_system()` / `turn_reply_user()` | `app/prompts.py` | `generate_turn_reply()` |
| `TurnReply` | `app/llm.py` | return type of `generate_turn_reply()`; consumed by `run_pipeline()`'s return dict (`reply`, `context_summary`, `changes_summary`, `is_question`) |

Removed in this merge (no longer exist): `_run_direct_answer()` in `app/pipeline.py`,
`llm.generate_answer()`, `prompts.GENERATE_ANSWER_SYSTEM` / `generate_answer_user()`,
`llm.TurnSummary` / `generate_turn_summary()`, `prompts.TURN_SUMMARY_SYSTEM` /
`turn_summary_user()`, and the SSE `token` event.
