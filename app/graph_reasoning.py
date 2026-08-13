"""Graph Reasoning — Phase 14. A small set of precomputed, queryable
questions over KnowledgeNode + KnowledgeEdge — not free-form graph
traversal exposed to an LLM. Each answers directly from the graph, with
zero reads of ChatSession.messages, per the plan's whole point: the graph
alone is sufficient once Phases 4-9 exist.

Not wired into the live turn — same standing as every module since Phase 1.
See ontology/PHASE14_GRAPH_REASONING.md for what rooms_exceeding_budget
assumes about Project.Quotation.RoomLineItems, which nothing populates yet."""

import re
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel
from rapidfuzz import fuzz

from app import graph_store
from app.dependency_graph import find_dependents
from app.models import KnowledgeNode

_MATERIAL_MATCH_THRESHOLD = 85  # rapidfuzz 0-100 scale; stricter than context_builder's dedup — a material name match is a direct answer, not a same-turn heuristic

_AMOUNT_RE = re.compile(r"[\d,]+(?:\.\d+)?")


def _parse_amount(value) -> Optional[float]:
    """Best-effort numeric extraction — a plain number as-is, or the first
    numeric run in a short string like "$8k"/"$8,000" ("k" immediately
    after the number treated as *1000). NOT a robust currency parser (none
    exists anywhere in this codebase); anything that doesn't parse cleanly
    returns None and the caller skips it rather than guessing."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = _AMOUNT_RE.search(value)
    if not match:
        return None
    number = float(match.group().replace(",", ""))
    if "k" in value[match.end() : match.end() + 2].lower():
        number *= 1000
    return number


class RoomOverage(BaseModel):
    room_id: str
    room_type: Optional[str] = None
    budget: float
    spent: float
    overage: float


class RoomRef(BaseModel):
    room_id: str
    room_type: Optional[str] = None


async def rooms_exceeding_budget(project_id: str) -> list[RoomOverage]:
    """A room "exceeds budget" when its Project.Rooms.<id>.Budget is less
    than the sum of Project.Quotation.RoomLineItems.<instance>.Amount rows
    tagged with that room's room_id (see PHASE14_GRAPH_REASONING.md — a
    RoomLineItems instance isn't path-nested under its room, it's linked via
    KnowledgeNode.room_id metadata). Rooms with no budget, no line items, or
    a budget/amount that doesn't parse as a number are skipped, not guessed
    at — nothing populates RoomLineItems yet, so this is exercised with
    seeded test data, same as app.dependency_graph's worked example."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    budgets = {n.room_id: n.value for n in nodes if n.node_type == "Budget" and n.room_id}
    room_types = {n.room_id: n.value for n in nodes if n.node_type == "RoomType" and n.room_id}

    line_items: dict[str, list[float]] = defaultdict(list)
    for n in nodes:
        if n.node_type == "Amount" and n.room_id:
            amount = _parse_amount(n.value)
            if amount is not None:
                line_items[n.room_id].append(amount)

    overages: list[RoomOverage] = []
    for room_id, raw_budget in budgets.items():
        budget = _parse_amount(raw_budget)
        if budget is None or room_id not in line_items:
            continue
        spent = sum(line_items[room_id])
        if spent > budget:
            overages.append(RoomOverage(room_id=room_id, room_type=room_types.get(room_id), budget=budget, spent=spent, overage=spent - budget))
    return overages


async def rooms_with_material(project_id: str, material: str) -> list[RoomRef]:
    """Every room with a Materials.<item>.Material leaf fuzzy-matching
    `material`, one RoomRef per room (not per matching material leaf)."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    room_types = {n.room_id: n.value for n in nodes if n.node_type == "RoomType" and n.room_id}

    matched_room_ids: list[str] = []
    for n in nodes:
        if n.node_type != "Material" or n.value is None or not n.room_id:
            continue
        if n.room_id in matched_room_ids:
            continue
        if fuzz.partial_ratio(str(n.value).lower(), material.lower()) >= _MATERIAL_MATCH_THRESHOLD:
            matched_room_ids.append(n.room_id)

    return [RoomRef(room_id=room_id, room_type=room_types.get(room_id)) for room_id in matched_room_ids]


async def items_depending_on(project_id: str, material_node_id: str) -> list[KnowledgeNode]:
    """Thin, purpose-named wrapper over app.dependency_graph.find_dependents
    (Phase 9) — "what depends on this material" is exactly "what has a
    derives_from edge targeting it"."""
    return await find_dependents(material_node_id, project_id, relation="derives_from")
