"""Exercises the actual FastAPI /chat route (not just the graph directly), so a
wiring bug in chat.py's signature/imports fails the suite instead of only
surfacing at runtime."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.test_graph import (
    fake_answer,
    fake_classify,
    fake_extract,
    fake_extract_graph_links,
    fake_gen_question,
    fake_generate_conflict_confirmation,
    fake_generate_wrapup_message,
    fake_query_catalog,
)


def test_chat_endpoint_streams_full_turn():
    with (
        patch("app.deepinfra.classify_intent", side_effect=fake_classify),
        patch("app.deepinfra.extract_fields", side_effect=fake_extract),
        patch("app.deepinfra.generate_question", side_effect=fake_gen_question),
        patch("app.deepinfra.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.deepinfra.generate_answer", side_effect=fake_answer),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
    ):
        from app.main import app

        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "I want a modern living room"})
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

            state_resp = client.get(f"/sessions/{session_id}/state")
            assert state_resp.status_code == 200
            data = state_resp.json()
            style_nodes = [n for n in data["knowledge_nodes"] if n["node_type"] == "Style"]
            assert style_nodes[0]["value"] == "modern"
            assert len(data["messages"]) >= 3
            assert any("Got it" in m["content"] for m in data["messages"] if m["role"] == "assistant")


def test_chat_endpoint_streams_confirm_change_event_for_a_critical_field_restatement():
    with (
        patch("app.deepinfra.classify_intent", side_effect=fake_classify),
        patch("app.deepinfra.extract_fields", side_effect=fake_extract),
        patch("app.deepinfra.generate_question", side_effect=fake_gen_question),
        patch("app.deepinfra.generate_wrapup_message", side_effect=fake_generate_wrapup_message),
        patch("app.deepinfra.generate_answer", side_effect=fake_answer),
        patch("app.deepinfra.generate_conflict_confirmation", side_effect=fake_generate_conflict_confirmation),
        patch("app.rag.query_catalog", side_effect=fake_query_catalog),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
    ):
        from app.deepinfra import ExtractedFields
        from app.main import app

        with TestClient(app) as client:
            with patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$15k")):
                resp1 = client.post("/chat", json={"message": "renovation, budget is $15k"})
            session_id = resp1.text.split('"session_id": "')[1].split('"')[0]

            with patch("app.deepinfra.extract_fields", side_effect=lambda message, known, **kwargs: ExtractedFields(overallBudget="$20k")):
                resp2 = client.post("/chat", json={"session_id": session_id, "message": "actually the budget is $20k"})

            assert resp2.status_code == 200
            body = resp2.text
            assert '"type": "confirm_change"' in body
            assert '"type": "ask_question"' not in body


async def fake_classify_always_update_context(message, history="", pending_field=None, **kwargs):
    return ["update_context"]


async def fake_generate_conflict_confirmation_delete(old_value, **kwargs):
    return f"Remove {old_value}?"


def test_chat_endpoint_streams_confirm_change_event_for_a_room_deletion():
    """Regression test: app.graph.delete_context_node's room-cascade
    confirmation dict was missing the `field_label` key app.chat.run_chat_turn
    reads to build the confirm_change SSE event — a KeyError that only
    surfaced hitting the real /chat endpoint (delete_context_node's own
    return value was tested in isolation in tests/test_delete_context.py, but
    nothing exercised chat.py's transport layer for that path). This drives
    a real room creation, then a real deletion request, through the actual
    HTTP route end to end."""
    from app.deepinfra import ExtractedFields

    with (
        patch("app.deepinfra.classify_intent", side_effect=fake_classify_always_update_context),
        patch("app.deepinfra.generate_question", side_effect=fake_gen_question),
        patch("app.deepinfra.extract_graph_links", side_effect=fake_extract_graph_links),
        patch("app.deepinfra.generate_conflict_confirmation_delete", side_effect=fake_generate_conflict_confirmation_delete),
        patch("app.database.connect_to_mongo"),
        patch("app.database.close_mongo_connection"),
    ):
        from app.main import app

        with TestClient(app) as client:
            with patch(
                "app.deepinfra.extract_fields",
                side_effect=lambda message, known, **kwargs: ExtractedFields(roomType="living room", style="modern"),
            ):
                resp1 = client.post("/chat", json={"message": "modern living room"})
            session_id = resp1.text.split('"session_id": "')[1].split('"')[0]

            resp2 = client.post("/chat", json={"session_id": session_id, "message": "remove the living room"})

            assert resp2.status_code == 200
            body = resp2.text
            assert '"type": "error"' not in body, body
            assert '"type": "confirm_change"' in body
            assert '"field": "room"' in body
