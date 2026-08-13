import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import settings

logger = logging.getLogger(__name__)

driver: AsyncDriver | None = None

# Mirrors app/database.py's _ensure_indexes for the graph side. No vector
# index here (unlike CatalogItem's Atlas $vectorSearch) — canonical_mapper's
# candidate pool is one project's freeform facts at a time, small and
# bounded by nature (see that module's own docstring), so a native vector
# index would add infrastructure with no query it actually speeds up today.
_CONSTRAINTS = [
    "CREATE CONSTRAINT knode_id IF NOT EXISTS FOR (n:KNode) REQUIRE n.node_id IS UNIQUE",
    "CREATE CONSTRAINT kversion_unique IF NOT EXISTS FOR (v:KNodeVersion) REQUIRE (v.node_id, v.version) IS UNIQUE",
    "CREATE INDEX knode_project_path IF NOT EXISTS FOR (n:KNode) ON (n.project_id, n.canonical_path)",
    "CREATE INDEX knode_project_lifecycle IF NOT EXISTS FOR (n:KNode) ON (n.project_id, n.lifecycle)",
    "CREATE INDEX knode_room IF NOT EXISTS FOR (n:KNode) ON (n.room_id)",
    "CREATE INDEX kversion_node IF NOT EXISTS FOR (v:KNodeVersion) ON (v.node_id)",
]


async def connect_to_neo4j() -> None:
    global driver
    driver = AsyncGraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    # No driver.verify_connectivity() here — it checks the driver's default
    # database ("neo4j"), which doesn't have to exist (e.g. a Neo4j Desktop
    # instance with only a custom-named database, like settings.neo4j_database
    # here). _ensure_constraints() below opens a session scoped to the actual
    # configured database, which validates connectivity against the database
    # this app will actually use, not some other one.
    await _ensure_constraints()


async def close_neo4j_connection() -> None:
    global driver
    if driver is not None:
        await driver.close()
        driver = None


async def _ensure_constraints() -> None:
    async with session() as s:
        for statement in _CONSTRAINTS:
            await s.run(statement)
    logger.info("Neo4j constraints/indexes ensured on database %s", settings.neo4j_database)


def session():
    if driver is None:
        raise RuntimeError("Neo4j driver not initialized — call connect_to_neo4j() first")
    return driver.session(database=settings.neo4j_database)
