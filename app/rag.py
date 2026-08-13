from app.llm import embed
from app.models import CatalogItem


async def query_catalog(query: str, limit: int = 5, style_tags: list[str] | None = None) -> list[CatalogItem]:
    [query_vector] = await embed([query])

    vector_stage: dict = {
        "$vectorSearch": {
            "index": "catalog_vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": limit * 20,
            "limit": limit,
        }
    }
    if style_tags:
        vector_stage["$vectorSearch"]["filter"] = {"style_tags": {"$in": style_tags}}

    # Beanie 2.0.0's aggregate().to_list() awaits collection.aggregate(...), but
    # this pymongo/motor version already returns the cursor synchronously —
    # bypass Beanie's wrapper and drive the motor collection directly.
    collection = CatalogItem.get_pymongo_collection()
    cursor = collection.aggregate([vector_stage])
    docs = await cursor.to_list(length=limit)
    return [CatalogItem.model_validate(doc) for doc in docs]
