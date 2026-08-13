




from __future__ import annotations

from app import context_builder
from app.neo4j_db import connect_to_neo4j, close_neo4j_connection
import asyncio

async def test_render_project_tree_text():
    await connect_to_neo4j()
    try:
        projectid = "ca43f14c-f6eb-4b7c-9725-7797f0cc5755"
        tree_text = await context_builder.render_project_tree_text(projectid)
        print(tree_text)
    finally:
        await close_neo4j_connection()

if __name__ == "__main__":
    asyncio.run(test_render_project_tree_text())