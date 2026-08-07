from fastapi import APIRouter, HTTPException

from app.models import ChatSession, KnowledgeNode, ProjectContext

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
    nodes = await KnowledgeNode.find(KnowledgeNode.project_id == session.project_id).to_list()
    return {
        "session_id": session.session_id,
        "project_id": session.project_id,
        "status": session.status,
        "active_room_id": session.active_room_id,
        "field_attempts": session.field_attempts,
        "pending_gap": session.pending_gap,
        "pending_confirmation": session.pending_confirmation,
        "knowledge_nodes": [n.model_dump() for n in nodes],
        "trace": [t.model_dump() for t in session.trace],
        "messages": [m.model_dump() for m in session.messages],
    }
