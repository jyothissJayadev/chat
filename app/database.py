import logging
from datetime import datetime, timezone

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, IndexModel
from pymongo.errors import OperationFailure

from app.config import settings
from app.models import CatalogItem, ChatSession, ProjectContext

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient | None = None


async def connect_to_mongo() -> None:
    global client
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db_name]

    # KnowledgeNode/KnowledgeNodeVersion/KnowledgeEdge (the knowledge graph)
    # live in Neo4j now — see app/neo4j_db.py and app/graph_store.py. Mongo
    # only holds dialogue mechanics and the catalog from here on.
    await init_beanie(database=db, document_models=[ChatSession, ProjectContext, CatalogItem])

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
