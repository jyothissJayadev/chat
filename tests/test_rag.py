from unittest.mock import AsyncMock, MagicMock, patch

from app.rag import query_catalog


async def test_query_catalog_builds_vector_search_pipeline():
    fake_vector = [0.1, 0.2, 0.3]

    fake_cursor = MagicMock()
    fake_cursor.to_list = AsyncMock(return_value=[])

    fake_collection = MagicMock()
    fake_collection.aggregate = MagicMock(return_value=fake_cursor)

    with (
        patch("app.rag.embed", new=AsyncMock(return_value=[fake_vector])),
        patch("app.models.CatalogItem.get_pymongo_collection", return_value=fake_collection),
    ):
        result = await query_catalog("modern living room", limit=3, style_tags=["modern"])

    assert result == []
    args, _ = fake_collection.aggregate.call_args
    pipeline = args[0]
    stage = pipeline[0]["$vectorSearch"]

    assert stage["index"] == "catalog_vector_index"
    assert stage["queryVector"] == fake_vector
    assert stage["limit"] == 3
    assert stage["filter"] == {"style_tags": {"$in": ["modern"]}}
    fake_cursor.to_list.assert_awaited_once_with(length=3)


async def test_query_catalog_converts_docs_to_catalog_items():
    fake_doc = {
        "_id": "507f1f77bcf86cd799439011",
        "item_id": "abc123",
        "title": "Modern Sofa",
        "description": "A sleek modern sofa",
        "style_tags": ["modern"],
        "embedding": [0.1, 0.2],
    }
    fake_cursor = MagicMock()
    fake_cursor.to_list = AsyncMock(return_value=[fake_doc])
    fake_collection = MagicMock()
    fake_collection.aggregate = MagicMock(return_value=fake_cursor)

    with (
        patch("app.rag.embed", new=AsyncMock(return_value=[[0.1]])),
        patch("app.models.CatalogItem.get_pymongo_collection", return_value=fake_collection),
    ):
        result = await query_catalog("sofa")

    assert len(result) == 1
    assert result[0].title == "Modern Sofa"
    assert result[0].item_id == "abc123"
