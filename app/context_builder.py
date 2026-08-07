"""Context Builder Pipeline — Phase 7. The live write path for every project
fact — app/graph.py's build_context_node calls build_context() below instead
of the old extract_fields_node + update_context_graph_node pair (which wrote
PartialContext/ContextGraph; both are gone, see ARCHITECTURE_BASELINE.md).
Structured fields are written directly; anything freeform goes through
Phase 5's canonical mapper.

See ontology/PHASE7_CONTEXT_BUILDER.md for the full design writeup —
notably: the tier-based conflict-resolution decision (now live —
BuildResult.pending_confirmations drives app.graph's confirm_conflict_node),
the same-turn dedup heuristic that enacts Gap 1 (no parallel freeform node
for a fact extract_fields already captured structurally), and two things
still NOT done as of the live cutover: relationship edges aren't persisted
(no KnowledgeEdge writer wired in yet — Phase 9's model exists, nothing
calls it from a live turn) and retract_node requests aren't honored
(KnowledgeNode has no active/superseded/retracted lifecycle yet — an
accepted, documented temporary regression from the old ContextGraph
behavior, not hacked around here)."""

import asyncio
import logging
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel
from rapidfuzz import fuzz

from app import canonical_mapper, deepinfra
from app.canonical_mapper import map_to_canonical
from app.models import FIELD_TIERS, ChangedBy, KnowledgeNode, room_type_matches, utcnow
from app.versioning import record_version

logger = logging.getLogger(__name__)

Tier = Literal["critical", "moderate", "optional"]

# Leaf node_type -> the FIELD_TIERS key that governs its conflict tier.
# Materials leaves (Label/Material/Specification) aren't listed individually
# here — FIELD_TIERS itself only has one "materials" entry covering the
# whole list, applied uniformly below. Public (not `_`-prefixed): also used
# by app/graph.py for question labels, confirmation prompts, and assumption
# summaries.
LEAF_TO_FIELD_NAME = {
    "ProjectType": "projectType",
    "Total": "overallBudget",
    "Value": "timeline",
    "RoomType": "roomType",
    "Budget": "budgetOrRequirement",
    "Style": "style",
    "SquareFootage": "squareFootage",
    "ExistingFurniture": "existingFurniture",
}

# extract_graph_links' old GraphNodeType vocabulary -> a map_to_canonical
# node_type_hint. "attribute" and "entity" are deliberately absent — they
# were the old catch-all types; passing a hint here would recreate exactly
# the mis-classification Gap 3's embedding-based inference exists to fix.
# "room" is deliberately absent too — a freeform room-typed proposal is
# dropped outright (see _is_room_duplicate below and Gap 1).
_OLD_TYPE_TO_HINT = {"preference": "ClientPreferences", "constraint": "Constraints"}

_DEDUP_SCORE_THRESHOLD = 80  # rapidfuzz 0-100 scale; same tool app/graph.py already uses for candidate search


class ProposedWrite(BaseModel):
    canonical_path: str
    node_type: str
    value: Any
    room_id: Optional[str] = None
    tier: Tier = "moderate"
    # Everything build_context() proposes today comes from the client's own
    # message (structured extraction or freeform mapping) — see
    # ontology/PHASE8_VERSIONING.md for why "inferred"/"system_default"
    # aren't produced by any call site yet.
    changed_by: ChangedBy = "user_message"


class Conflict(BaseModel):
    canonical_path: str
    node_type: str
    old_value: Any
    new_value: Any  # None when kind="delete" — see app.graph.delete_context_node
    tier: Tier
    # "edit" is build_context's own conflict kind (unchanged); "delete" and
    # "delete_room" are produced by app.graph.delete_context_node (a single
    # node vs. a whole room's subtree, respectively) for a critical-tier
    # removal, reusing this same model/confirmation machinery rather than a
    # second parallel type — see app.graph.confirm_conflict_node's branch.
    kind: Literal["edit", "delete", "delete_room"] = "edit"


class BuildResult(BaseModel):
    project_id: str
    room_id: Optional[str] = None
    written: list[str] = []
    pending_confirmations: list[Conflict] = []
    # Extracted but not persisted — see module docstring. Present so a
    # caller/test can observe what extract_relationships found even though
    # nothing acts on it yet.
    freeform_relationships: list[dict] = []


async def extract_entities(message: str, known: dict, *, capture: dict | None = None):
    """Thin rename of deepinfra.extract_fields for this module's own
    vocabulary — no new extraction logic, same model call."""
    return await deepinfra.extract_fields(message, known, capture=capture)


async def extract_relationships(
    message: str, anchors: list[dict], candidate_nodes: list[dict], recent_edges: list[dict], *, capture: dict | None = None
):
    """Thin rename of deepinfra.extract_graph_links. candidate_nodes/recent_edges
    are always passed empty by build_context (see its docstring) — revise_node/
    retract_node targeting is out of scope for this phase either way, so
    there's no candidate list worth showing the model."""
    return await deepinfra.extract_graph_links(message, anchors, candidate_nodes, recent_edges, capture=capture)


async def _existing_rooms(project_id: str) -> dict[str, str]:
    leaves = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.node_type == "RoomType", KnowledgeNode.lifecycle == "active"
    ).to_list()
    return {leaf.room_id: str(leaf.value) for leaf in leaves if leaf.room_id and leaf.value is not None}


async def _resolve_room(project_id: str, room_type: Optional[str], active_room_id: Optional[str]) -> Optional[str]:
    """Returns the room_id this turn's room-scoped facts belong to, fuzzy-
    matching room_type against existing rooms (room_type_matches,
    app/models.py) or creating a new Rooms instance when nothing matches.
    Falls back to active_room_id when no room_type was stated this turn."""
    if room_type:
        for room_id, existing_type in (await _existing_rooms(project_id)).items():
            if room_type_matches(existing_type, room_type):
                return room_id
        new_room_id = uuid4().hex[:8]
        await apply_to_graph(
            project_id,
            [ProposedWrite(canonical_path=f"Project.Rooms.{new_room_id}.RoomType", node_type="RoomType", value=room_type, room_id=new_room_id, tier="critical")],
        )
        return new_room_id
    return active_room_id


async def known_fields(project_id: str, room_id: Optional[str]) -> dict[str, Any]:
    """Project- and room-scoped known values, in the same shape
    extract_fields'/generate_question's/generate_answer's prompts already
    expect. Public (not `_`-prefixed): app/graph.py reuses this directly for
    every node that needs to show "what's known so far" to a model —
    generate_question_node, decline_field_node, generate_answer_node,
    analyze_context_node — rather than each re-deriving it."""
    nodes = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.lifecycle == "active"
    ).to_list()
    known: dict[str, Any] = {}
    for node in nodes:
        if node.value is None:
            continue
        if node.canonical_path == "Project.BasicInformation.ProjectType":
            known["projectType"] = node.value
        elif node.canonical_path == "Project.Budget.Total":
            known["overallBudget"] = node.value
        elif node.canonical_path == "Project.Timeline.Value":
            known["timeline"] = node.value
        elif room_id and node.room_id == room_id:
            field_name = LEAF_TO_FIELD_NAME.get(node.node_type)
            if field_name:
                known[field_name] = node.value
    return known


async def materialize_project_summary(project_id: str) -> dict[str, Any]:
    """Flat, read-side snapshot of a completed project's structured
    fields — {"projectType": ..., "overallBudget": ..., "timeline": ...,
    "rooms": [{"roomType": ..., "materials": [...]}, ...]} — the mirror
    image of build_context's write side. Used by
    app.graph.complete_project_node to populate ProjectContext.summary."""
    nodes = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.lifecycle == "active"
    ).to_list()
    by_path = {n.canonical_path: n for n in nodes}

    def value_at(path: str) -> Any:
        node = by_path.get(path)
        return node.value if node else None

    summary: dict[str, Any] = {
        "projectType": value_at("Project.BasicInformation.ProjectType"),
        "overallBudget": value_at("Project.Budget.Total"),
        "timeline": value_at("Project.Timeline.Value"),
    }

    room_ids = sorted({n.room_id for n in nodes if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms" and n.room_id})
    rooms = []
    for room_id in room_ids:
        room_prefix = f"Project.Rooms.{room_id}"
        room: dict[str, Any] = {
            "room_id": room_id,
            "roomType": value_at(f"{room_prefix}.RoomType"),
            "budgetOrRequirement": value_at(f"{room_prefix}.Budget"),
            "style": value_at(f"{room_prefix}.Style"),
            "squareFootage": value_at(f"{room_prefix}.SquareFootage"),
            "existingFurniture": value_at(f"{room_prefix}.ExistingFurniture"),
        }
        materials_prefix = f"{room_prefix}.Materials."
        item_slugs = sorted(
            {p[len(materials_prefix):].split(".")[0] for p in by_path if p.startswith(materials_prefix)}
        )
        room["materials"] = [
            {
                "item": value_at(f"{materials_prefix}{slug}.Label"),
                "material": value_at(f"{materials_prefix}{slug}.Material"),
                "specification": value_at(f"{materials_prefix}{slug}.Specification"),
            }
            for slug in item_slugs
        ]
        rooms.append(room)
    summary["rooms"] = rooms
    return summary


async def _build_anchors(project_id: str) -> list[dict]:
    """id/label/type anchor dicts for extract_relationships' prompt (the
    same shape the old, now-deleted _build_anchor_scaffold produced),
    sourced from KnowledgeNode — see that function's own docstring for why
    candidate nodes/edges are always empty regardless."""
    nodes = await KnowledgeNode.find(
        KnowledgeNode.project_id == project_id, KnowledgeNode.lifecycle == "active"
    ).to_list()
    anchors = [{"id": "project", "label": "Project", "type": "project"}]

    leaves_by_room: dict[str, dict[str, KnowledgeNode]] = {}
    for node in nodes:
        if node.room_id and node.node_type in ("RoomType", "Budget"):
            leaves_by_room.setdefault(node.room_id, {})[node.node_type] = node

    total = next((n for n in nodes if n.canonical_path == "Project.Budget.Total"), None)
    if total is not None:
        anchors.append({"id": "budget:total", "label": f"Overall budget: {total.value}", "type": "budget"})

    for node in nodes:
        if node.node_type != "Rooms" or node.canonical_path == "Project.Rooms":
            continue
        room_id = node.canonical_path.rsplit(".", 1)[-1]
        leaves = leaves_by_room.get(room_id, {})
        label = str(leaves["RoomType"].value) if "RoomType" in leaves else f"Room {room_id}"
        anchors.append({"id": f"room:{room_id}", "label": label, "type": "room"})
        if "Budget" in leaves:
            anchors.append({"id": f"budget:{room_id}", "label": f"Budget: {leaves['Budget'].value}", "type": "budget"})

    return anchors


def _is_duplicate_of_structured_value(label: str, structured_values: list[str]) -> bool:
    """Gap 1's enactment: a freeform node whose label is basically restating
    a value extract_entities already captured structurally THIS SAME TURN
    (e.g. a "preference" node for "modern" when style="modern" was also just
    extracted) must not also become a separate KnowledgeNode. Same-turn only
    — this can't catch a freeform mention of something structurally captured
    on an EARLIER turn, since that value isn't in scope here; map_to_canonical's
    own alias/embedding matching is what would need to catch that case, and
    Style/Budget are deliberately outside its scope (see app/canonical_mapper.py)."""
    return any(fuzz.partial_ratio(label.lower(), value.lower()) >= _DEDUP_SCORE_THRESHOLD for value in structured_values if value)


async def detect_conflicts(proposed: list[ProposedWrite], project_id: str) -> tuple[list[ProposedWrite], list[Conflict]]:
    """Tier-based conflict resolution (see PHASE7_CONTEXT_BUILDER.md): a
    proposed write conflicts only when a node already exists at its
    canonical_path with a DIFFERENT, non-null value AND the field's tier is
    "critical" — moderate/optional fields, and anything with no prior value,
    apply automatically."""
    applyable: list[ProposedWrite] = []
    conflicts: list[Conflict] = []
    for write in proposed:
        existing = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == write.canonical_path)
        if existing is None or existing.value is None or existing.value == write.value:
            applyable.append(write)
            continue
        if write.tier == "critical":
            conflicts.append(
                Conflict(canonical_path=write.canonical_path, node_type=write.node_type, old_value=existing.value, new_value=write.value, tier=write.tier)
            )
        else:
            applyable.append(write)
    return applyable, conflicts


async def apply_to_graph(project_id: str, writes: list[ProposedWrite]) -> list[str]:
    """Writes each ProposedWrite to its canonical leaf, creating any missing
    ancestor container along the way (canonical_mapper.ensure_path). A new
    leaf, or an existing one whose value actually changes, gets its `version`
    bumped (or set to 1) AND an append-only KnowledgeNodeVersion row via
    app.versioning.record_version — see ontology/PHASE8_VERSIONING.md, and is
    included in the returned list. A write whose value is identical to
    what's already stored is a true no-op: no version bump, no history row,
    and NOT included in the returned list — build_context_node's "Got it —
    noted ..." summary is built from this list, so a same-turn restatement
    of an already-known fact (the extraction model doesn't always limit
    itself to what's newly stated, despite being asked to) doesn't pad out
    that message with facts that didn't actually change."""
    written: list[str] = []
    for write in writes:
        parent_path = write.canonical_path.rsplit(".", 1)[0]
        parent = await canonical_mapper.ensure_path(parent_path, project_id)
        existing = await KnowledgeNode.find_one(KnowledgeNode.project_id == project_id, KnowledgeNode.canonical_path == write.canonical_path)
        if existing is None:
            node = KnowledgeNode(
                canonical_path=write.canonical_path, node_type=write.node_type, parent_id=parent.node_id,
                value=write.value, project_id=project_id, room_id=write.room_id, changed_by=write.changed_by,
            )
            await node.insert()
            parent.children_ids.append(node.node_id)
            await parent.save()
            await record_version(node, changed_by=write.changed_by)
            written.append(write.canonical_path)
        elif existing.value != write.value:
            existing.value = write.value
            existing.changed_by = write.changed_by
            existing.version += 1
            existing.updated_at = utcnow()
            await existing.save()
            await record_version(existing, changed_by=write.changed_by)
            written.append(write.canonical_path)
    return written


async def build_context(
    message: str,
    project_id: str,
    active_room_id: Optional[str] = None,
    known_override: Optional[dict] = None,
    room_hint: Optional[str] = None,
) -> BuildResult:
    """room_hint (app.tasks.TaskSpec.room_hint, from a segmented multi-room
    clause — see app.understanding.segment_clauses) is a raw room-TYPE NAME,
    not a room_id — tried through the exact same _resolve_room fuzzy-match-
    or-create path as extracted.roomType, just lower priority: the clause's
    own extraction usually already names its room directly (that's how
    segmentation found the boundary in the first place), so room_hint mainly
    covers a trailing fragment clause ("and a tv") that doesn't restate it."""
    known = {**(await known_fields(project_id, active_room_id)), **(known_override or {})}
    anchors = await _build_anchors(project_id)

    entities_result, relationships_result = await asyncio.gather(
        extract_entities(message, known), extract_relationships(message, anchors, [], []), return_exceptions=True
    )

    if isinstance(entities_result, Exception):
        logger.warning("build_context: extract_entities failed for project %s (%s: %s) — skipping structured extraction this turn", project_id, type(entities_result).__name__, entities_result)
        extracted = None
    else:
        extracted = entities_result

    if isinstance(relationships_result, Exception):
        logger.warning("build_context: extract_relationships failed for project %s (%s: %s) — skipping freeform extraction this turn", project_id, type(relationships_result).__name__, relationships_result)
        graph_extraction = None
    else:
        graph_extraction, _status = relationships_result

    room_id = active_room_id
    proposed: list[ProposedWrite] = []
    structured_values: list[str] = []

    def _add(field_name: str, canonical_path: str, node_type: str, value: Any, r_id: Optional[str] = None) -> None:
        if value is None:
            return
        proposed.append(
            ProposedWrite(canonical_path=canonical_path, node_type=node_type, value=value, room_id=r_id, tier=FIELD_TIERS.get(field_name, "moderate"))
        )
        if isinstance(value, str):
            structured_values.append(value)

    if extracted is not None:
        room_id = await _resolve_room(project_id, extracted.roomType or room_hint, active_room_id)
        has_room_scoped_fields = extracted.materials or any(
            v is not None for v in (extracted.budgetOrRequirement, extracted.style, extracted.squareFootage, extracted.existingFurniture)
        )
        if room_id is None and has_room_scoped_fields:
            # Room-scoped facts with no explicit roomType and no active room
            # to attach to (e.g. "make it modern" as the very first message)
            # still need SOME room to belong to — same as the pre-cutover
            # PartialContext.resolve_room(None) behavior: create an
            # unlabeled room rather than silently dropping the data. The
            # room's own container is created lazily by apply_to_graph's
            # ensure_path when the first field below is written.
            room_id = uuid4().hex[:8]

        if extracted.projectType is not None:
            _add("projectType", "Project.BasicInformation.ProjectType", "ProjectType", extracted.projectType)
        if extracted.overallBudget is not None:
            _add("overallBudget", "Project.Budget.Total", "Total", extracted.overallBudget)
        if extracted.timeline is not None:
            _add("timeline", "Project.Timeline.Value", "Value", extracted.timeline)

        if room_id is not None:
            if extracted.roomType is not None:
                _add("roomType", f"Project.Rooms.{room_id}.RoomType", "RoomType", extracted.roomType, room_id)
            if extracted.budgetOrRequirement is not None:
                _add("budgetOrRequirement", f"Project.Rooms.{room_id}.Budget", "Budget", extracted.budgetOrRequirement, room_id)
            if extracted.style is not None:
                _add("style", f"Project.Rooms.{room_id}.Style", "Style", extracted.style, room_id)
            if extracted.squareFootage is not None:
                _add("squareFootage", f"Project.Rooms.{room_id}.SquareFootage", "SquareFootage", extracted.squareFootage, room_id)
            if extracted.existingFurniture is not None:
                _add("existingFurniture", f"Project.Rooms.{room_id}.ExistingFurniture", "ExistingFurniture", extracted.existingFurniture, room_id)

            for material in extracted.materials or []:
                item_path = f"Project.Rooms.{room_id}.Materials.{canonical_mapper.slugify(material.item)}"
                _add("materials", f"{item_path}.Label", "Label", material.item, room_id)
                _add("materials", f"{item_path}.Material", "Material", material.material, room_id)
                if material.specification is not None:
                    _add("materials", f"{item_path}.Specification", "Specification", material.specification, room_id)

        for extra in extracted.additionalRoomBudgets or []:
            if extra.roomType is None or extra.budgetOrRequirement is None:
                continue
            extra_room_id = await _resolve_room(project_id, extra.roomType, None)
            _add("budgetOrRequirement", f"Project.Rooms.{extra_room_id}.Budget", "Budget", extra.budgetOrRequirement, extra_room_id)

        for room_name in extracted.mentionedAdditionalRooms or []:
            if room_name is not None:
                await _resolve_room(project_id, room_name, None)

    applyable, conflicts = await detect_conflicts(proposed, project_id)
    written = await apply_to_graph(project_id, applyable)

    freeform_relationships: list[dict] = []
    if graph_extraction is not None:
        freeform_relationships = [{"source": e.source, "target": e.target, "relation": e.relation} for e in graph_extraction.new_edges]

        for new_node in graph_extraction.new_nodes:
            if new_node.type == "room":
                continue
            if _is_duplicate_of_structured_value(new_node.label, structured_values):
                continue
            hint = _OLD_TYPE_TO_HINT.get(new_node.type)
            match = await map_to_canonical(new_node.label, hint, project_id, room_id)
            written.append(match.canonical_path)

        if graph_extraction.retracted_node_ids:
            logger.warning(
                "build_context: %d retraction(s) requested for project %s but not applied — "
                "KnowledgeNode has no active/superseded/retracted lifecycle yet, see ontology/PHASE7_CONTEXT_BUILDER.md",
                len(graph_extraction.retracted_node_ids), project_id,
            )

    return BuildResult(project_id=project_id, room_id=room_id, written=written, pending_confirmations=conflicts, freeform_relationships=freeform_relationships)
