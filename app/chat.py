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
    session_id: Optional[str], message: str, image_url: Optional[str] = None
) -> AsyncIterator[dict]:
    """Runs one full chat turn end-to-end and yields transport-agnostic event
    dicts, formatted onto the wire as SSE by the /chat endpoint below.

    Event shapes:
      {"type": "image_description", "content": str}
      {"type": "token", "content": str}
      {"type": "progress", "node": str}
      {"type": "trace", "node": str, "entries": list[dict]}  # TraceEntry.model_dump() per entry
      {"type": "context_updated", "message": str}
      {"type": "confirm_change", "field": str, "old_value": Any, "new_value": Any, "question": str}
      {"type": "ask_question", "question": str}
      {"type": "wrapup", "message": str}
      {"type": "error", "message": str}
      {"type": "done", "session_id": str, "status": str}
      {"type": "heartbeat"}
    """
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
            "active_room_id": session.active_room_id,
            "field_attempts": dict(session.field_attempts),
            "intent": [],
            "tasks": [],
            "retrieved": "",
            "pending_question": None,
            "pending_gap": session.pending_gap,
            "pending_confirmation": session.pending_confirmation,
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
                    yield {"type": "token", "content": chunk["token"]}
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

        session.messages.append(Message(role="user", content=message))
        if final_state["answer"]:
            session.messages.append(Message(role="assistant", content=final_state["answer"]))

        session.active_room_id = final_state.get("active_room_id", session.active_room_id)
        session.field_attempts = final_state.get("field_attempts", session.field_attempts)
        session.trace.extend(final_state["trace"])
        # validate_completeness_node's verdict, computed earlier in this same
        # turn — equivalent to recomputing find_knowledge_gap() now (no node
        # downstream of it mutates project state), just without redoing the work.
        session.status = "complete" if final_state.get("complete") else "in_progress"

        # Confirm what was captured before moving on to the next question, so
        # an update_context turn doesn't just silently jump to a new question
        # with no acknowledgment of what the user just said.
        update_summary = final_state.get("update_summary")
        if update_summary:
            session.messages.append(Message(role="assistant", content=update_summary))
            yield {"type": "context_updated", "message": update_summary}

        # The "type 2" closing statement (see generate_wrapup_message) — set
        # exactly when complete_project_node ran this turn, mutually
        # exclusive with pending_confirmation/pending_question below.
        wrapup_message = final_state.get("wrapup_message")
        if wrapup_message:
            session.messages.append(Message(role="assistant", content=wrapup_message))
            yield {"type": "wrapup", "message": wrapup_message}

        # A critical-tier value change held back by build_context_node — see
        # app.graph.confirm_conflict_node. Mutually exclusive with
        # pending_question (build_context_node never sets both the same
        # turn — see its docstring).
        pending_confirmation = final_state.get("pending_confirmation")
        pending_question = final_state.get("pending_question")
        if pending_confirmation and session.status == "in_progress":
            session.messages.append(Message(role="assistant", content=pending_confirmation["question"]))
            session.pending_confirmation = pending_confirmation
            session.pending_gap = None
            yield {
                "type": "confirm_change",
                "field": pending_confirmation["field_label"],
                "old_value": pending_confirmation["old_value"],
                "new_value": pending_confirmation["new_value"],
                "question": pending_confirmation["question"],
            }
        elif pending_question and session.status == "in_progress":
            session.messages.append(Message(role="assistant", content=pending_question))
            session.pending_gap = final_state.get("pending_gap")
            session.pending_confirmation = None
            yield {"type": "ask_question", "question": pending_question}
        else:
            session.pending_gap = None
            session.pending_confirmation = None

        await save_session(session)

        root_span.update(output=final_state.get("answer") or pending_question or wrapup_message or "")
        yield {"type": "done", "session_id": session.session_id, "status": session.status}


@router.post("/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        async for event in run_chat_turn(req.session_id, req.message, req.image_url):
            if event["type"] == "heartbeat":
                yield ": keep-alive\n\n"
                continue
            payload = {k: v for k, v in event.items() if k != "type"}
            yield _sse(event["type"], payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
