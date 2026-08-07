import os

os.environ.setdefault("DEEPINFRA_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")

# Explicit assignment (not setdefault) — real env vars take priority over the
# .env file in pydantic-settings, so this reliably blanks out whatever real
# Langfuse project is configured in .env, ensuring the test suite never sends
# traces to a live project regardless of what's in the developer's local .env.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

import pytest_asyncio
from beanie import init_beanie
from mongomock_motor import AsyncMongoMockClient

from app.models import CatalogItem, ChatSession, KnowledgeEdge, KnowledgeNode, KnowledgeNodeVersion, ProjectContext


@pytest_asyncio.fixture(autouse=True, scope="session")
async def init_test_db():
    client = AsyncMongoMockClient()
    await init_beanie(
        database=client["test_db"],
        document_models=[ChatSession, ProjectContext, CatalogItem, KnowledgeNode, KnowledgeNodeVersion, KnowledgeEdge],
    )
