# Interior Design Chat

FastAPI + MongoDB Atlas + LangGraph + DeepInfra chat backend for interior design
project intake, with a built-in debug viewer for inspecting session state.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `DEEP_INFRA_API` — from your DeepInfra account
   - `MONGO_URI` — a MongoDB **Atlas** connection string (required for Atlas
     Vector Search used by the catalog/RAG feature; a local `mongod` won't work)
   - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` —
     optional. Leave blank to disable tracing entirely; nothing else changes.
2. Install dependencies:
   ```
   python -m venv .venv
   source .venv/Scripts/activate  # or .venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   uvicorn app.main:app --reload
   ```
4. Check `/health`, then open `/viewer?session_id=<id>` after your first `/chat`
   call to watch context, model trace, and conversation fill in live.

## Chat transport

`POST /chat` drives a chat turn via `app.chat.run_chat_turn` and streams the
result back as Server-Sent Events (`text/event-stream`), used by the debug
viewer (`app/static/viewer.html`) and any other client. One request per turn:
leave `session_id` empty on the first message, then pass the `session_id`
from that turn's `done` event on subsequent messages to continue the same
session. Event types on the stream: `token`, `progress`, `context_updated`,
`ask_question`, `wrapup`, `image_description`, `error`, `done`, plus a
`: keep-alive` comment line as a heartbeat during long-running turns.

A previous version of this backend also exposed a persistent gRPC stream
(`app/grpc_server.py`) for a planned Node.js backend integration. That was
removed — DeepInfra's own client-facing surface doesn't support gRPC, and a
per-turn SSE reconnect is negligible overhead since session state lives in
MongoDB, not the connection.

## Model routing (DeepInfra)

| Role | Model |
|---|---|
| classify_intent / extract_fields / update_context_graph / generate_question / generate_wrapup_message | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| generate_answer | `deepseek-ai/DeepSeek-V4-Pro` |
| vision | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| embed | `Qwen/Qwen3-Embedding-8B` |

### Two question "types"

The intake flow generates two distinct kinds of assistant turns, both driven
by `app/deepinfra.py` but with different intents:

- **Ask** (`generate_question`) — a genuine slot-filling question that needs
  a data answer, used while `PartialContext.next_field_to_ask()` still has
  something open.
- **Wrap-up** (`generate_wrapup_message`) — a declarative closing statement,
  not a question, generated once by `save_project_node` when the intake is
  fully done. A field the user explicitly declined and exhausted its retry
  budget on (status `"skipped"`) does **not** get one more re-ask before
  wrapping up — `PartialContext.is_complete()` treats it as resolved as soon
  as nothing else is open, and `save_project_node` fills it in with
  `infer_missing_field` (marked `"assumed"`) instead. This avoids an extra,
  redundant round-trip once the flow has effectively nothing left to
  ask — see `GraphState.question_generated` in `app/graph.py`, the flag every
  node that produces an ask/wrap-up sets so `analyze_context_node` (which
  runs on every branch) never generates a second one for the same turn.

extract/question-gen roles were originally on GLM-4.7-Flash, then
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` — both turned out slow, for
different reasons. GLM-4.7-Flash is a reasoning model that burned 8-23s per
call on invisible chain-of-thought before emitting output. Mistral-Small
isn't a reasoning model, but was live-timed at 11-62s per call regardless of
output length, vs. consistently <2-3s for every "Turbo"-branded model tried
— almost certainly served on a cold/shared DeepInfra tier rather than a
low-latency one. Reusing the same Turbo model already proven fast for
classify_intent (confirmed working with instructor's TOOLS mode too) fixed
both extraction roles and question-gen at once.

`extract_fields` and `update_context_graph` run **sequentially**, not
concurrently, despite what an earlier version of this doc claimed — see the
comment on that edge in `app/graph.py`: a prior attempt at LangGraph's
dynamic fan-out had a race where `validate_completeness` could run before
`extract_fields`' state update was merged, silently dropping the turn's
extraction ~1/3 of the time. Correctness over the ~3s it would save.

## Observability (Langfuse)

If `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, every `/chat` call
produces one `chat_turn` trace with each graph node as a nested span and every
underlying LLM call auto-captured as a generation (real token usage, cost,
latency) — see `app/observability.py`. This is the primary tool for diagnosing
per-turn latency; the in-app viewer's Process Trace panel shows the same node
timings for quick local debugging without needing Langfuse running.

## Tests

```
pytest
```

`test_graph.py` mocks DeepInfra calls to verify intent routing, field
extraction/merging, and the leftover-question loop surviving an unrelated
turn. `test_rag.py` mocks the embedding call to verify the vector search
aggregation pipeline shape. `test_chat.py` covers turn logic through the
real HTTP/SSE endpoint.
"# chat" 
