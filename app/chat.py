import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langfuse import get_client
from pydantic import BaseModel

from app.database import get_or_create_session, save_session
from app.graph import app_graph, describe_image_node
from app.models import Message

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    image_url: Optional[str] = None
    # {op_id: chosen_value} answering a prior turn's operation_questions
    # event (see app.graph.classify_intent_node's resume branch) — the only
    # structured (non-message-text) reply channel this endpoint has.
    operation_answers: Optional[dict[str, str]] = None
    # {op_id: [room_id, ...]} — set ONLY for an op_id whose chosen answer was
    # a bundled multi-room option (app.llm.RoomResolutionOption.room_ids).
    # operation_answers[op_id] still carries that option's display label;
    # this is the parallel structured data app.graph.classify_intent_node's
    # resume branch uses to fan that op out into one write task per room.
    operation_room_selections: Optional[dict[str, list[str]]] = None


# Some branches (e.g. update_context: extract_fields -> update_context_graph ->
# generate_question) run several sequential reasoning-model calls with zero
# bytes sent to the client in between — long enough that browsers/proxies can
# drop the connection as idle, which looks like a silent hang. Emit a
# "heartbeat" event if the graph goes quiet for longer than this; the SSE
# endpoint below turns it into a comment line to keep the connection alive.
HEARTBEAT_SECONDS = 8

_SENTINEL = object()


def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


async def run_chat_turn(
    session_id: Optional[str],
    message: str,
    image_url: Optional[str] = None,
    operation_answers: Optional[dict[str, str]] = None,
    operation_room_selections: Optional[dict[str, list[str]]] = None,
) -> AsyncIterator[dict]:
    """Runs one full chat turn end-to-end and yields transport-agnostic event
    dicts, formatted onto the wire as SSE by the /chat endpoint below.

    Event shapes:
      {"type": "image_description", "content": str}
      {"type": "progress", "node": str}
      {"type": "trace", "node": str, "entries": list[dict]}  # TraceEntry.model_dump() per entry
      {"type": "operation_progress", "stage": str, "ok": bool}
      {"type": "pipeline_result", "context_summary": str|None, "changes_summary": str|None, "reply": str, "is_question": bool}
      {"type": "operation_questions", "questions": list[dict]}
      # [{"op_id", "text", "question", "options": [{"id", "label", "room_ids"}, ...], "allow_custom": true}, ...]
      # The viewer must render both the option list AND a free-text input for
      # each question — `allow_custom` is always true (the reply channel
      # below accepts any string per op_id, an option's own label or
      # something the user typed). An option with "room_ids" set (2+ ids) is
      # a bundled multi-room choice (app.llm.RoomResolutionOption) — picking
      # it must send BOTH operation_answers[op_id]=that option's label AND
      # operation_room_selections[op_id]=that option's room_ids on the next
      # call, so the turn applies to every bundled room.
      {"type": "error", "message": str}
      {"type": "done", "session_id": str, "status": str}
      {"type": "heartbeat"}

    `operation_progress` fires once per pipeline stage as it completes (see
    app.pipeline.run_pipeline: first_action / context_retrieval /
    database_query / summary, whichever apply this turn) — finer-grained
    than the per-graph-node `progress` event above (which only fires once
    for the whole `run_pipeline` node).

    `pipeline_result` is the single join-step reply for this turn — replaces
    the old separate context_updated/ask_question/wrapup events, and is no
    longer split across a streamed `token` answer plus a separate closing
    line (see app.llm.generate_turn_reply — one LLM call now produces the
    whole thing, not streamed token-by-token). `reply` is the turn's whole
    conversational text, already ending on either the next question
    (is_question=true) or a closing/completion line (is_question=false).

    `operation_questions` fires when classify_intent_node couldn't ground
    one or more write operations to a graph location (see the
    classifier-connection plan) — reply on the NEXT call with
    `operation_answers={op_id: chosen_value}` for each op_id listed, plus
    `operation_room_selections={op_id: [room_id, ...]}` for any op_id whose
    chosen value was a bundled multi-room option."""
    session = await get_or_create_session(session_id)
    history = "\n".join(f"{m.role}: {m.content}" for m in session.messages[-10:])
    graph_message = message

    langfuse = get_client()

    with langfuse.start_as_current_observation(
        name="chat_turn",
        as_type="span",
        input=message,
        metadata={"session_id": session.session_id, "has_image": bool(image_url)},
    ) as root_span:
        if image_url:
            entry, description = await describe_image_node({"message": graph_message}, image_url)
            session.trace.append(entry)
            yield {"type": "trace", "node": "describe_image", "entries": [entry.model_dump(mode="json")]}
            graph_message = f"{graph_message}\n\n[Image description: {description}]"
            yield {"type": "image_description", "content": description}

        state = {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "message": graph_message,
            "history": history,
            "skipped_rooms": list(session.skipped_rooms),
            "project_type_skipped": session.project_type_skipped,
            "current_field": session.current_field,
            "active_room_id": session.active_room_id,
            "intent": [],
            "tasks": [],
            "pending_operation_questions": session.pending_operation_questions,
            "operation_answers": operation_answers,
            "operation_room_selections": operation_room_selections,
            "pending_gap": session.pending_gap,
            "reply": "",
            "context_summary": None,
            "changes_summary": None,
            "is_question": False,
            "complete": False,
            "trace": [],
        }

        queue: asyncio.Queue = asyncio.Queue()

        async def run_graph():
            try:
                async for mode, chunk in app_graph.astream(state, stream_mode=["custom", "updates", "values"]):
                    await queue.put((mode, chunk))
            except Exception as exc:
                logger.exception("graph execution failed for session %s", session.session_id)
                await queue.put(("error", str(exc)))
            finally:
                await queue.put((_SENTINEL, None))

        task = asyncio.create_task(run_graph())
        final_state = None
        graph_error = None

        # NOT asyncio.wait_for(queue.get(), timeout=...) — wait_for cancels
        # the underlying get() on timeout, and asyncio.Queue can drop an
        # item that was handed to a get() right as it gets cancelled. Under
        # real request timing (not reproducible in a quick local script)
        # this silently lost extract_fields' output on ~2/3 of turns whose
        # node happened to straddle the heartbeat interval. asyncio.wait on
        # a persistent, never-cancelled future avoids that race entirely.
        get_future = asyncio.ensure_future(queue.get())
        try:
            while True:
                done, _ = await asyncio.wait({get_future}, timeout=HEARTBEAT_SECONDS)
                if not done:
                    yield {"type": "heartbeat"}
                    continue

                mode, chunk = get_future.result()
                get_future = asyncio.ensure_future(queue.get())

                if mode is _SENTINEL:
                    break
                if mode == "custom":
                    # app.pipeline.run_pipeline's per-stage progress events
                    # (chunk["type"] == "operation_progress") — the only
                    # custom-stream producer now that generate_turn_reply
                    # replaces the old streamed direct-answer call.
                    if chunk.get("type") == "operation_progress":
                        yield {
                            "type": "operation_progress",
                            "stage": chunk["stage"],
                            "ok": chunk["ok"],
                        }
                elif mode == "updates":
                    for node_name, update in chunk.items():
                        yield {"type": "progress", "node": node_name}
                        entries = (update or {}).get("trace") or []
                        if entries:
                            yield {
                                "type": "trace",
                                "node": node_name,
                                "entries": [e.model_dump(mode="json") for e in entries],
                            }
                elif mode == "values":
                    final_state = chunk
                elif mode == "error":
                    graph_error = chunk
        finally:
            await task
            if not get_future.done():
                get_future.cancel()

        if graph_error or final_state is None:
            root_span.update(
                output=None, level="ERROR", status_message=graph_error or "no result"
            )
            yield {"type": "error", "message": graph_error or "graph did not produce a result"}
            return

        # An operation_answers resume sends message="" (see ChatRequest's
        # docstring — the reply travels structurally, not as message text),
        # so fall back to the chosen value(s) for the persisted transcript;
        # otherwise a reload/loadSession shows a blank user bubble where the
        # picked option should read (e.g. "Kitchen").
        user_message_content = message or (", ".join(operation_answers.values()) if operation_answers else message)
        session.messages.append(Message(role="user", content=user_message_content))

        session.skipped_rooms = final_state.get("skipped_rooms", session.skipped_rooms)
        session.project_type_skipped = final_state.get("project_type_skipped", session.project_type_skipped)
        session.current_field = final_state.get("current_field", session.current_field)
        session.active_room_id = final_state.get("active_room_id", session.active_room_id)
        session.trace.extend(final_state["trace"])
        # app.pipeline.run_pipeline's own verdict (question_engine.
        # find_knowledge_gaps returned None) — nothing downstream mutates
        # project state, so no need to recompute it here.
        session.status = "complete" if final_state.get("complete") else "in_progress"

        # The pipeline's single join-step reply — replaces the old separate
        # context_updated/ask_question/wrapup events, and the old separate
        # streamed DIRECT_ANSWER message. `reply` is either the next
        # question (is_question=true) or a closing/completion line
        # (is_question=false); either way it's the one thing to show the
        # user this turn.
        reply = final_state.get("reply")
        is_question = final_state.get("is_question", False)
        if reply:
            context_summary = final_state.get("context_summary")
            changes_summary = final_state.get("changes_summary")
            session.messages.append(
                Message(role="assistant", content=reply, context_summary=context_summary, changes_summary=changes_summary)
            )
            yield {
                "type": "pipeline_result",
                "context_summary": context_summary,
                "changes_summary": changes_summary,
                "reply": reply,
                "is_question": is_question,
            }

        # A held-back unresolved-connection batch (classify_intent_node) and
        # an ordinary knowledge-gap question (run_pipeline) are mutually
        # exclusive by construction — pending_operation_questions only ever
        # comes from the short-circuit branch that never reaches
        # run_pipeline, so reply is always empty on that branch.
        pending_operation_questions = final_state.get("pending_operation_questions")
        if pending_operation_questions and session.status == "in_progress":
            for q in pending_operation_questions["questions"]:
                session.messages.append(Message(role="assistant", content=q["question"]))
            session.pending_operation_questions = pending_operation_questions
            session.pending_gap = None
            yield {"type": "operation_questions", "questions": pending_operation_questions["questions"]}
        elif is_question and session.status == "in_progress":
            session.pending_gap = final_state.get("pending_gap")
            session.pending_operation_questions = None
        else:
            session.pending_gap = None
            session.pending_operation_questions = None

        await save_session(session)

        root_span.update(output=reply or "")
        yield {"type": "done", "session_id": session.session_id, "status": session.status}


@router.post("/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in run_chat_turn(
            req.session_id, req.message, req.image_url, req.operation_answers, req.operation_room_selections
        ):
            if event["type"] == "heartbeat":
                yield ": keep-alive\n\n"
                continue
            payload = {k: v for k, v in event.items() if k != "type"}
            yield _sse(event["type"], payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
