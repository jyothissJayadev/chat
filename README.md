# Interior Design Chat

FastAPI + MongoDB Atlas + LangGraph + Fireworks AI chat backend for interior design
project intake, with a built-in debug viewer for inspecting session state.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `LLM_API_KEY` — from your LLM provider account (currently Fireworks AI; see
     [Model routing](#model-routing-fireworks-ai) below — `app/llm.py`'s client/field
     names are deliberately provider-generic, so switching providers again later is a
     config change, not a rename sweep across the codebase)
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
removed — the LLM provider's own OpenAI-compatible client-facing surface
doesn't support gRPC, and a per-turn SSE reconnect is negligible overhead
since session state lives in MongoDB, not the connection.

## Model routing (Fireworks AI)

All calls go through one `AsyncOpenAI` client in `app/llm.py`, pointed at
Fireworks AI (`app/config.py`'s `llm_base_url`/`llm_api_key`) — was DeepInfra
before a provider switch on 2026-08-13. Field/module names are deliberately
provider-generic, so a future switch is a config change, not a rename sweep.

| Role | Model |
|---|---|
| classify_operations (intent + multi-operation split + graph connection) | `accounts/fireworks/models/gpt-oss-120b` |
| extract_fields / extract_graph_links / generate_question / generate_wrapup_message | `accounts/fireworks/models/gpt-oss-120b` |
| generate_answer | `accounts/fireworks/models/gpt-oss-120b` |
| vision | `accounts/fireworks/models/qwen3p7-plus` |
| embed | `accounts/fireworks/models/qwen3-embedding-8b` |

### Two question "types"

The intake flow generates two distinct kinds of assistant turns, both driven
by `app/llm.py` but with different intents:

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

All on DeepInfra at the time: extract/question-gen roles were originally on
GLM-4.7-Flash, then `mistralai/Mistral-Small-3.2-24B-Instruct-2506` — both
turned out slow, for different reasons. GLM-4.7-Flash is a reasoning model
that burned 8-23s per call on invisible chain-of-thought before emitting
output. Mistral-Small isn't a reasoning model, but was live-timed at 11-62s
per call regardless of output length, vs. consistently <2-3s for every
"Turbo"-branded model tried — almost certainly served on a cold/shared
DeepInfra tier rather than a low-latency one. Reusing the same Turbo model
already proven fast on the intent-classification role at the time (confirmed
working with instructor's TOOLS mode too) fixed both extraction roles and
question-gen at once. The classifier role itself later moved to a larger
model — `Llama-3.3-70B-Instruct-Turbo`, then `gpt-oss-120b` — since splitting
a message into several correctly-scoped, graph-grounded operations is a
harder task than the single-label classification this history is about.
After the Fireworks switch every text/reasoning role (including extraction/
question-gen) was consolidated onto the one `gpt-oss-120b` deployment (see
the routing table above) — it's itself a reasoning model, so the same
latency caution this paragraph documents is worth re-checking live rather
than assumed fixed.

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

`test_graph.py` mocks LLM calls to verify intent routing, field
extraction/merging, and the leftover-question loop surviving an unrelated
turn. `test_rag.py` mocks the embedding call to verify the vector search
aggregation pipeline shape. `test_chat.py` covers turn logic through the
real HTTP/SSE endpoint.
"# chat" 
