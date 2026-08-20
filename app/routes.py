from fastapi import APIRouter, HTTPException

from app import graph_store
from app.models import ChatSession, ProjectContext

router = APIRouter()


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    sessions = await ChatSession.find_all().sort(-ChatSession.updated_at).limit(limit).to_list()
    return [
        {
            "session_id": s.session_id,
            "status": s.status,
            "updated_at": s.updated_at,
            "message_count": len(s.messages),
        }
        for s in sessions
    ]


@router.get("/projects")
async def list_projects(limit: int = 20):
    return await ProjectContext.find_all().sort(-ProjectContext.created_at).limit(limit).to_list()


@router.get("/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    session = await ChatSession.find_one(ChatSession.session_id == session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    nodes = await graph_store.find_nodes(session.project_id)
    return {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "status": session.status,
        "skipped_rooms": session.skipped_rooms,
        "project_type_skipped": session.project_type_skipped,
        "current_field": session.current_field,
        "pending_gap": session.pending_gap,
        "pending_operation_questions": session.pending_operation_questions,
        "knowledge_nodes": [n.model_dump() for n in nodes],
        "trace": [t.model_dump() for t in session.trace],
        "messages": [m.model_dump() for m in session.messages],
    }
