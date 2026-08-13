import os

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
# The knowledge graph lives in Neo4j now (app/graph_store.py) — there's no
# in-memory Cypher-compatible fake equivalent to mongomock_motor below, so
# tests run against a real local instance. Defaults here match the Neo4j
# Desktop instance already configured in .env; override via real env vars
# (e.g. in CI, pointed at a throwaway container) to run against a different
# one without touching this file.
os.environ.setdefault("NEO4J_URI", "neo4j://127.0.0.1:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "11111111")
os.environ.setdefault("NEO4J_DATABASE", "velocitychat")

# Explicit assignment (not setdefault) — real env vars take priority over the
# .env file in pydantic-settings, so this reliably blanks out whatever real
# Langfuse project is configured in .env, ensuring the test suite never sends
# traces to a live project regardless of what's in the developer's local .env.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

import pytest_asyncio
from beanie import init_beanie
from mongomock_motor import AsyncMongoMockClient

from app import graph_store, neo4j_db
from app.models import CatalogItem, ChatSession, ProjectContext


@pytest_asyncio.fixture(autouse=True, scope="session")
async def init_test_db():
    client = AsyncMongoMockClient()
    await init_beanie(database=client["test_db"], document_models=[ChatSession, ProjectContext, CatalogItem])

    await neo4j_db.connect_to_neo4j()
    yield
    await neo4j_db.close_neo4j_connection()


@pytest_asyncio.fixture(autouse=True)
async def clean_graph(monkeypatch):
    """velocitychat (NEO4J_DATABASE in .env) is the developer's real Neo4j
    Desktop database, not a throwaway — there's no in-memory Cypher-compatible
    fake the way mongomock_motor gives Mongo a fresh database per session, so
    a blanket `MATCH (n) DETACH DELETE n` before every test was deleting every
    real project's graph data on each test run. Instead, track exactly which
    project_ids THIS test writes via graph_store.insert_node (every write
    path funnels through it — canonical_mapper/context_builder/versioning all
    call `graph_store.insert_node`, never construct :KNode Cypher directly)
    and delete only those project_ids' nodes once the test finishes — real
    dev data under any other project_id is never touched. Runs as teardown
    (after `yield`), not setup, so it also cleans up a test's own leftovers
    even when the test raises. `x.node_id IN ids` (not `x:KNode`) so this
    also removes each deleted node's :KNodeVersion rows, which carry the same
    node_id but aren't graph-linked to it (see app/graph_store.py) — :REL
    edges need no separate handling since DETACH DELETE drops them with their
    endpoint node."""
    touched_project_ids: set[str] = set()
    original_insert_node = graph_store.insert_node

    async def tracking_insert_node(node):
        touched_project_ids.add(node.project_id)
        return await original_insert_node(node)

    monkeypatch.setattr(graph_store, "insert_node", tracking_insert_node)
    yield
    if touched_project_ids:
        async with neo4j_db.session() as s:
            await s.run(
                """
                MATCH (n:KNode) WHERE n.project_id IN $project_ids
                WITH collect(DISTINCT n.node_id) AS ids
                MATCH (x) WHERE x.node_id IN ids
                DETACH DELETE x
                """,
                project_ids=list(touched_project_ids),
            )
