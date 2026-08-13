"""Exercises the actual FastAPI /chat route (not just the graph directly), so a
wiring bug in chat.py's signature/imports fails the suite instead of only
surfacing at runtime.

Uses httpx.AsyncClient(transport=ASGITransport(...)) rather than
starlette.testclient.TestClient — TestClient drives the ASGI app through its
own internal event loop (a sync-to-async bridge), separate from the one
pytest-asyncio's session-scoped fixtures (and the shared Neo4j driver they
create, see tests/conftest.py) run on; awaiting that driver from TestClient's
loop raises "attached to a different loop". Staying on one async client keeps
everything on the single session-scoped loop the whole suite already shares."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.test_graph import (
    fake_answer,
    fake_classify_operations,
    fake_extract,
    fake_extract_graph_links,
    fake_gen_question,
    fake_generate_conflict_confirmation,
    fake_generate_wrapup_message,
    fake_query_catalog,
    fake_resolve_room_connections,
)


@pytest.mark.asyncio
async def test_chat_endpoint_streams_full_turn():
    with (
        patch("app.llm.classify_operations", side_effect=fake_classify_operations),
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.generate_question", side_effect=fake_gen_question),
        patch("app.llm.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.llm.generate_answer", side_effect=fake_answer),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
        # tests/conftest.py already owns one shared Neo4j driver for the
        # whole session — letting the app's own lifespan close it on
        # shutdown would null out app.neo4j_db.driver for every test that
        # runs after this one. Same reasoning as the Mongo patches above.
        patch("app.neo4j_db.connect_to_neo4j"),
        patch("app.neo4j_db.close_neo4j_connection"),
    ):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "I want a modern living room"})
            assert resp.status_code == 200
            body = resp.text
            assert '"type": "progress"' in body
            assert '"type": "context_updated"' in body
            assert '"type": "ask_question"' in body
            assert '"type": "done"' in body
            # confirmation must precede the next question, not the other way round
            assert body.index('"type": "context_updated"') < body.index('"type": "ask_question"')
            assert "style: modern" in body

            session_id = body.split('"session_id": "')[1].split('"')[0]

            state_resp = await client.get(f"/sessions/{session_id}/state")
            assert state_resp.status_code == 200
            data = state_resp.json()
            style_nodes = [n for n in data["knowledge_nodes"] if n["node_type"] == "Style"]
            assert style_nodes[0]["value"] == "modern"
            assert len(data["messages"]) >= 3
            assert any("Got it" in m["content"] for m in data["messages"] if m["role"] == "assistant")


@pytest.mark.asyncio
async def test_chat_endpoint_streams_confirm_change_event_for_a_critical_field_restatement():
    with (
        patch("app.llm.classify_operations", side_effect=fake_classify_operations),
        patch("app.llm.extract_fields", side_effect=fake_extract),
        patch("app.llm.generate_question", side_effect=fake_gen_question),
        patch("app.llm.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.llm.generate_answer", side_effect=fake_answer),
        patch("app.llm.generate_conflict_confirmation", side_effect=fake_generate_conflict_confirmation),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
        patch("app.neo4j_db.connect_to_neo4j"),
        patch("app.neo4j_db.close_neo4j_connection"),
    ):
        from app.llm import ExtractedFields
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$15k")):
                resp1 = await client.post("/chat", json={"message": "renovation, budget is $15k"})
            session_id = resp1.text.split('"session_id": "')[1].split('"')[0]

            with patch("app.llm.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")):
                resp2 = await client.post("/chat", json={"session_id": session_id, "message": "actually the budget is $20k"})

            assert resp2.status_code == 200
            body = resp2.text
            assert '"type": "confirm_change"' in body
            assert '"type": "ask_question"' not in body


@pytest.mark.asyncio
async def test_chat_endpoint_round_trips_operation_questions_then_applies_the_answer():
    """A multi-op turn where one write op's connection comes back null holds
    the WHOLE batch for clarification (see app.graph.classify_intent_node) —
    nothing writes on turn 1. Turn 2 sends operation_answers; the previously
    withheld batch resumes and commits, with no re-classification call and
    no repeated operation_questions event."""

    async def fixed_classify(message, history="", pending_field=None, *, tree_text=None, capture=None):
        from app.llm import Operation

        return [
            Operation(id="op_1", text="the overall budget is $30k", intent="CONTEXT_UPDATE", connection="Project"),
            Operation(id="op_2", text="the cabinet should be walnut", intent="CONTEXT_UPDATE", connection=None),
        ]

    async def fake_extract_dispatch(message, known, **kwargs):
        from app.llm import ExtractedFields

        if "budget" in message.lower():
            return ExtractedFields(overallBudget="$30k")
        return ExtractedFields()

    reclassify_calls = 0

    async def counting_classify(*args, **kwargs):
        nonlocal reclassify_calls
        reclassify_calls += 1
        return await fixed_classify(*args, **kwargs)

    with (
        patch("app.llm.classify_operations", side_effect=counting_classify),
        patch("app.llm.extract_fields", side_effect=fake_extract_dispatch),
        patch("app.llm.generate_question", side_effect=fake_gen_question),
        patch("app.llm.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.llm.generate_answer", side_effect=fake_answer),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
        patch("app.neo4j_db.connect_to_neo4j"),
        patch("app.neo4j_db.close_neo4j_connection"),
    ):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp1 = await client.post(
                "/chat", json={"message": "the overall budget is $30k and the cabinet should be walnut"}
            )
            assert resp1.status_code == 200
            body1 = resp1.text
            assert '"type": "operation_questions"' in body1
            assert '"op_2"' in body1
            assert reclassify_calls == 1

            session_id = body1.split('"session_id": "')[1].split('"')[0]
            state_resp1 = await client.get(f"/sessions/{session_id}/state")
            assert state_resp1.json()["knowledge_nodes"] == [], "nothing should write while a clarification is pending"

            resp2 = await client.post(
                "/chat",
                json={
                    "session_id": session_id,
                    "message": "",
                    "operation_answers": {"op_2": "Kids Bedroom"},
                },
            )
            assert resp2.status_code == 200
            body2 = resp2.text
            assert '"type": "operation_questions"' not in body2
            assert reclassify_calls == 1, "the resume turn must not call classify_operations again"

            state_resp2 = await client.get(f"/sessions/{session_id}/state")
            nodes = state_resp2.json()["knowledge_nodes"]
            budget_nodes = [n for n in nodes if n["node_type"] == "Total"]
            assert budget_nodes and budget_nodes[0]["value"] == "$30k"


async def fake_classify_update_or_delete(message, history="", pending_field=None, **kwargs):
    """Stands in for classify_operations across a create-then-remove turn
    pair — RULE 7 of the real prompt classifies a removal statement as
    CONTEXT_DELETE directly (no separate guard_delete override needed
    anymore, unlike the old classify_intent + guard_delete pipeline this
    replaces). Both turns in this pair name "living room" explicitly, so
    connection grounds to it directly — app.graph.classify_intent_node holds
    the turn unconditionally whenever connection is None, so an ungrounded
    op here would never reach build_context/delete_context at all."""
    from app.llm import Operation

    intent = "CONTEXT_DELETE" if "remove" in message.lower() else "CONTEXT_UPDATE"
    connection = "Living Room" if "living room" in message.lower() else None
    return [Operation(text=message, intent=intent, connection=connection)]


async def fake_generate_conflict_confirmation_delete(old_value, **kwargs):
    return f"Remove {old_value}?"


@pytest.mark.asyncio
async def test_chat_endpoint_streams_confirm_change_event_for_a_room_deletion():
    """Regression test: app.graph.delete_context_node's room-cascade
    confirmation dict was missing the `field_label` key app.chat.run_chat_turn
    reads to build the confirm_change SSE event — a KeyError that only
    surfaced hitting the real /chat endpoint (delete_context_node's own
    return value was tested in isolation in tests/test_delete_context.py, but
    nothing exercised chat.py's transport layer for that path). This drives
    a real room creation, then a real deletion request, through the actual
    HTTP route end to end."""
    from app.llm import ExtractedFields

    with (
        patch("app.llm.classify_operations", side_effect=fake_classify_update_or_delete),
        patch("app.llm.generate_question", side_effect=fake_gen_question),
        patch("app.llm.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.llm.generate_conflict_confirmation_delete", side_effect=fake_generate_conflict_confirmation_delete),
        patch("app.llm.resolve_room_connections", side_effect=fake_resolve_room_connections),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
        patch("app.neo4j_db.connect_to_neo4j"),
        patch("app.neo4j_db.close_neo4j_connection"),
    ):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch(
                "app.llm.extract_fields",
                side_effect=lambda message, known, **kwargs: ExtractedFields(roomType="living room", style="modern"),
            ):
                resp1 = await client.post("/chat", json={"message": "modern living room"})
            session_id = resp1.text.split('"session_id": "')[1].split('"')[0]

            resp2 = await client.post("/chat", json={"session_id": session_id, "message": "remove the living room"})

            assert resp2.status_code == 200
            body = resp2.text
            assert '"type": "error"' not in body, body
            assert '"type": "confirm_change"' in body
            assert '"field": "room"' in body
