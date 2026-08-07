import logging
from datetime import datetime, timezone

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, IndexModel
from pymongo.errors import OperationFailure

from app.config import settings
from app.models import CatalogItem, ChatSession, KnowledgeEdge, KnowledgeNode, KnowledgeNodeVersion, ProjectContext

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient | None = None


async def connect_to_mongo() -> None:
    global client
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db_name]

    await init_beanie(
        database=db,
        document_models=[ChatSession, ProjectContext, CatalogItem, KnowledgeNode, KnowledgeNodeVersion, KnowledgeEdge],
    )

    await _ensure_indexes(db)


async def close_mongo_connection() -> None:
    if client is not None:
        client.close()


async def get_or_create_session(session_id: str | None) -> ChatSession:
    if session_id:
        existing = await ChatSession.find_one(ChatSession.session_id == session_id)
        if existing:
            return existing
        return await ChatSession(session_id=session_id).insert()
    return await ChatSession().insert()


async def save_session(session: ChatSession) -> None:
    session.updated_at = datetime.now(timezone.utc)
    await session.save()


async def _ensure_indexes(db) -> None:
    sessions = db["sessions"]
    await sessions.create_indexes(
        [
            IndexModel("session_id", unique=True),
            IndexModel(
                "updated_at",
                expireAfterSeconds=settings.session_ttl_days * 86400,
            ),
        ]
    )

    projects = db["projects"]
    await projects.create_indexes(
        [
            IndexModel("project_id", unique=True),
            IndexModel("session_id"),
        ]
    )

    catalog = db["catalog"]
    await catalog.create_indexes([IndexModel([("style_tags", ASCENDING)])])

    knowledge_nodes = db["knowledge_nodes"]
    await knowledge_nodes.create_indexes(
        [
            IndexModel("node_id", unique=True),
            IndexModel([("project_id", 1), ("canonical_path", 1)]),
            IndexModel([("parent_id", 1)]),
            # Unused today (single-tenant) — indexed now per Phase 6's own
            # rationale in app/models.py::KnowledgeNode.tenant_id.
            IndexModel("tenant_id"),
        ]
    )

    knowledge_node_versions = db["knowledge_node_versions"]
    await knowledge_node_versions.create_indexes([IndexModel([("node_id", 1), ("version", 1)])])

    knowledge_edges = db["knowledge_edges"]
    await knowledge_edges.create_indexes(
        [IndexModel([("project_id", 1), ("target_id", 1)]), IndexModel([("project_id", 1), ("source_id", 1)])]
    )

    try:
        existing = [idx["name"] async for idx in catalog.list_search_indexes()]
        if "catalog_vector_index" not in existing:
            await catalog.create_search_index(
                {
                    "name": "catalog_vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 4096,  # Qwen3-Embedding-8B native output size
                                "similarity": "cosine",
                            }
                        ]
                    },
                }
            )
            logger.info("Created Atlas Vector Search index 'catalog_vector_index'")
    except OperationFailure as exc:
        logger.warning(
            "Skipping Atlas Vector Search index creation (requires Atlas): %s", exc
        )
