"""Context Builder Pipeline — Phase 7. The live write path for every project
fact — app/graph.py's build_context_node calls build_context() below instead
of the old extract_fields_node + update_context_graph_node pair (which wrote
PartialContext/ContextGraph; both are gone, see ARCHITECTURE_BASELINE.md).
Structured fields are written directly; anything freeform goes through
Phase 5's canonical mapper.

See ontology/PHASE7_CONTEXT_BUILDER.md for the full design writeup —
notably the same-turn dedup heuristic that enacts Gap 1 (no parallel freeform node
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

from app import canonical_mapper, llm, graph_store
from app.canonical_mapper import CanonicalMatch, map_to_canonical
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


class BuildResult(BaseModel):
    project_id: str
    room_id: Optional[str] = None
    written: list[str] = []
    # Extracted but not persisted — see module docstring. Present so a
    # caller/test can observe what extract_relationships found even though
    # nothing acts on it yet.
    freeform_relationships: list[dict] = []


class RoomResolution(BaseModel):
    """_resolve_room's pure result: which room this turn's room-scoped facts
    belong to, and — when nothing matched an existing room — the type a NEW
    room needs its RoomType leaf written as. Deliberately carries no write of
    its own (see _resolve_room's docstring); resolve_context turns
    new_room_type into an ordinary ProposedWrite, same as every other
    structured field, rather than this function writing anything itself."""

    room_id: Optional[str] = None
    new_room_type: Optional[str] = None


class FreeformMention(BaseModel):
    """One extract_graph_links new_node, previewed (not yet mapped) — see
    ResolvedBuild. preview.canonical_path is the path this mention WOULD
    land at if committed right now (via map_to_canonical(commit=False));
    commit_context re-resolves it for real at commit time rather than
    trusting this preview verbatim, since project state can change between
    resolve and commit (see ResolvedBuild's docstring)."""

    raw_entity: str
    node_type_hint: Optional[str]
    preview: CanonicalMatch


class ResolvedBuild(BaseModel):
    """build_context's read/extraction phase, split out so a caller (e.g. a
    future clustering pass over one turn's several operations) can learn
    what a message WOULD write — target paths, room resolution, freeform
    mention previews — before any turn's actual field-value writes commit.
    Not a strict promise: map_to_canonical's preview for a freeform mention,
    and RoomResolution's new_room_type, are re-resolved for real inside
    commit_context rather than blindly trusted, so a resolved plan can never
    apply a stale decision if project state changed in between (e.g. another
    operation in the same turn already created the room/entity this one was
    about to)."""

    project_id: str
    room_id: Optional[str] = None
    room_resolution: RoomResolution
    proposed: list[ProposedWrite] = []
    freeform_mentions: list[FreeformMention] = []
    freeform_relationships: list[dict] = []


async def extract_entities(message: str, known: dict, *, capture: dict | None = None):
    """Thin rename of llm.extract_fields for this module's own
    vocabulary — no new extraction logic, same model call."""
    return await llm.extract_fields(message, known, capture=capture)


async def extract_relationships(
    message: str, anchors: list[dict], candidate_nodes: list[dict], recent_edges: list[dict], *, capture: dict | None = None
):
    """Thin rename of llm.extract_graph_links. candidate_nodes/recent_edges
    are always passed empty by build_context (see its docstring) — revise_node/
    retract_node targeting is out of scope for this phase either way, so
    there's no candidate list worth showing the model."""
    return await llm.extract_graph_links(message, anchors, candidate_nodes, recent_edges, capture=capture)


async def _existing_rooms(project_id: str) -> dict[str, str]:
    leaves = await graph_store.find_nodes(project_id, node_type="RoomType", lifecycle="active")
    return {leaf.room_id: str(leaf.value) for leaf in leaves if leaf.room_id and leaf.value is not None}


async def _resolve_room(project_id: str, room_type: Optional[str], fallback_room_id: Optional[str] = None) -> RoomResolution:
    """Pure lookup — fuzzy-matches room_type against existing rooms
    (room_type_matches, app/models.py) or allocates a fresh room_id for a
    new one, WITHOUT writing anything (unlike this function's old shape,
    which wrote the new room's RoomType leaf immediately via its own
    apply_to_graph call). Falls back to `fallback_room_id` when no room_type
    was stated this turn — this is the CALLER's own already-grounded room
    (resolve_context's own `room_id` param, sourced from task.connection),
    not a session-level "last active room" (there's no more active_room_id
    fallback — removed system-wide). A caller with no grounded room at all
    (fallback_room_id=None) relies on resolve_context's own "create an
    unlabeled room" fallback instead of this function inventing one.

    The caller is responsible for turning new_room_type (when set) into a
    ProposedWrite for the RoomType leaf — see resolve_context's
    _resolve_and_register_room, which folds it into the SAME proposed batch
    every other structured field goes through, rather than a separate write
    round-trip. This is what makes room resolution safe to call from a pure
    resolve phase (see ResolvedBuild)."""
    if room_type:
        for room_id, existing_type in (await _existing_rooms(project_id)).items():
            if room_type_matches(existing_type, room_type):
                return RoomResolution(room_id=room_id)
        return RoomResolution(room_id=uuid4().hex[:8], new_room_type=room_type)
    return RoomResolution(room_id=fallback_room_id)


async def known_fields(project_id: str, room_id: Optional[str]) -> dict[str, Any]:
    """Project- and room-scoped known values, in the same shape
    extract_fields'/generate_question's/generate_answer's prompts already
    expect. Public (not `_`-prefixed): app/graph.py reuses this directly for
    every node that needs to show "what's known so far" to a model —
    generate_question_node, decline_field_node, generate_answer_node,
    analyze_context_node — rather than each re-deriving it."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
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
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
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


# Every freeform instance's own possible leaf field names, in a fixed
# display order — see ontology/v1.yaml's per-type `fields` lists. Checked
# uniformly across all 5 freeform node types rather than keyed per-type:
# app.canonical_mapper._create_instance today only ever writes the Label
# leaf (Material/Specification/Notes aren't wired up by any write path yet),
# so this mostly renders "Label=..." alone — the other keys are included
# only if some future writer starts populating them.
_FREEFORM_LEAF_KEYS = ("Label", "Material", "Specification", "Notes")
# Instance node_types that can appear directly under a room (see
# ontology/v1.yaml's Rooms.children) — Unmapped is included here too since a
# room-scoped Unmapped instance (room_id set) reads more usefully inline
# with the room than off in the project-level Unmapped section below.
_ROOM_FREEFORM_TYPES = ("Materials", "Furniture", "Attributes", "Unmapped")


def _tree_branch(items: list[tuple[str, list[str]]]) -> list[str]:
    """Renders one level of ├──/└──/│ tree connectors for `items` (header,
    already-rendered detail lines). Detail lines may themselves be the
    output of a nested _tree_branch call — prefixing them with the parent
    connector's continuation string is what makes recursive nesting compose
    correctly without each call needing to know its own depth."""
    lines: list[str] = []
    for i, (header, detail) in enumerate(items):
        is_last = i == len(items) - 1
        lines.append(("└── " if is_last else "├── ") + header)
        cont = "    " if is_last else "│   "
        lines.extend(cont + line for line in detail)
    return lines


def _is_container(node: KnowledgeNode) -> bool:
    """True for an ancestor container node (e.g. "Project.Rooms.<id>.Materials"
    itself, node_type="Materials") rather than a real instance under it
    (e.g. "...Materials.countertop") — canonical_mapper.ensure_path gives a
    container the SAME node_type as the instances it holds (see
    _client_preference_instances in tests/test_context_builder.py for the
    same convention noted from the read side), so a container's last path
    segment is always its own node_type — a real instance's slug never is."""
    return node.canonical_path.rsplit(".", 1)[-1] == node.node_type


def _freeform_instance_line(instance: KnowledgeNode, by_path: dict[str, KnowledgeNode]) -> tuple[str, list[str]]:
    slug = instance.canonical_path.rsplit(".", 1)[-1]
    header = f"{instance.node_type}.{slug}"
    parts = []
    for key in _FREEFORM_LEAF_KEYS:
        leaf = by_path.get(f"{instance.canonical_path}.{key}")
        if leaf is not None and leaf.value is not None:
            parts.append(f'{key}="{leaf.value}"')
    return header, ([", ".join(parts)] if parts else [])


async def render_project_tree_text(project_id: str) -> str:
    """Plain-text, root-relative rendering of the live KnowledgeNode tree —
    fed into classify_operations' system prompt (see
    prompts.classify_operations_system) so the classifier can ground each
    operation's `connection` against the project's actual current state.

    Root-relative: no "Project." prefix on any path, matching the convention
    app.llm.Operation.connection itself uses (e.g. "Rooms.a1b2c3d4",
    not "Project.Rooms.a1b2c3d4"). Value-only: no node_id/version/confidence/
    status/lifecycle/tenant_id/embedding ever appears, only canonical
    structure + value — and a null/absent field is omitted entirely rather
    than shown as a blank, so a half-filled project reads as exactly what's
    known so far."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    print("this is the nodes from render_project_tree_text",nodes)
    by_path = {n.canonical_path: n for n in nodes}

    def value_at(path: str) -> Any:
        node = by_path.get(path)
        return node.value if node else None

    sections: list[tuple[str, list[str]]] = []

    project_type = value_at("Project.BasicInformation.ProjectType")
    if project_type is not None:
        sections.append(("BasicInformation", _tree_branch([(f'ProjectType = "{project_type}"', [])])))

    total = value_at("Project.Budget.Total")
    if total is not None:
        sections.append(("Budget", _tree_branch([(f'Total = "{total}"', [])])))

    timeline = value_at("Project.Timeline.Value")
    if timeline is not None:
        sections.append(("Timeline", _tree_branch([(f'Value = "{timeline}"', [])])))

    room_ids = sorted({n.room_id for n in nodes if n.node_type == "Rooms" and n.canonical_path != "Project.Rooms" and n.room_id})
    if room_ids:
        room_items: list[tuple[str, list[str]]] = []
        for room_id in room_ids:
            prefix = f"Project.Rooms.{room_id}"
            fields = [
                (field, value_at(f"{prefix}.{field}"))
                for field in ("RoomType", "Budget", "Style", "SquareFootage", "ExistingFurniture")
            ]
            instances = sorted(
                (n for n in nodes if n.node_type in _ROOM_FREEFORM_TYPES and n.room_id == room_id and not _is_container(n)),
                key=lambda n: (n.node_type, n.canonical_path),
            )
            # Fields and freeform instances are siblings under the room — one
            # flat _tree_branch call so every line (a leaf field or an
            # instance header) gets its own ├──/└── bullet together, matching
            # the sample tree's flat sibling layout rather than two visually
            # inconsistent groups.
            room_children = [(f'{field} = "{value}"', []) for field, value in fields if value is not None]
            room_children += [_freeform_instance_line(n, by_path) for n in instances]
            room_items.append((f"Rooms.{room_id}", _tree_branch(room_children)))
        sections.append(("Rooms", _tree_branch(room_items)))

    req_instances = sorted(
        (n for n in nodes if n.node_type in ("Constraints", "ClientPreferences") and not _is_container(n)),
        key=lambda n: (n.node_type, n.canonical_path),
    )
    if req_instances:
        sections.append(("Requirements", _tree_branch([_freeform_instance_line(n, by_path) for n in req_instances])))

    unmapped_instances = sorted(
        (n for n in nodes if n.node_type == "Unmapped" and n.room_id is None and not _is_container(n)),
        key=lambda n: n.canonical_path,
    )
    if unmapped_instances:
        sections.append(("Unmapped", _tree_branch([_freeform_instance_line(n, by_path) for n in unmapped_instances])))
    result = "\n".join(["Project", *_tree_branch(sections)])
    print("this is the result from  treee data",result)
    return result


async def _build_anchors(project_id: str) -> list[dict]:
    """id/label/type anchor dicts for extract_relationships' prompt (the
    same shape the old, now-deleted _build_anchor_scaffold produced),
    sourced from KnowledgeNode — see that function's own docstring for why
    candidate nodes/edges are always empty regardless."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
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
        existing = await graph_store.find_one(project_id, write.canonical_path)
        if existing is None:
            node = KnowledgeNode(
                canonical_path=write.canonical_path, node_type=write.node_type, parent_id=parent.node_id,
                value=write.value, project_id=project_id, room_id=write.room_id, changed_by=write.changed_by,
            )
            await graph_store.insert_node(node)
            await record_version(node, changed_by=write.changed_by)
            written.append(write.canonical_path)
        elif existing.value != write.value:
            existing.value = write.value
            existing.changed_by = write.changed_by
            existing.version += 1
            existing.updated_at = utcnow()
            await graph_store.save_node(existing)
            await record_version(existing, changed_by=write.changed_by)
            written.append(write.canonical_path)
    return written


async def resolve_context(
    message: str,
    project_id: str,
    room_id: Optional[str] = None,
    known_override: Optional[dict] = None,
    room_hint: Optional[str] = None,
) -> ResolvedBuild:
    """The read/extraction half of build_context, split out (see
    ResolvedBuild's docstring) so a caller can learn what a message WOULD
    write before anything actually commits. room_hint (app.tasks.TaskSpec.
    room_hint) is a raw room-TYPE NAME, not a room_id — tried through the
    exact same _resolve_room fuzzy-match-or-create path as extracted.roomType,
    just lower priority: the clause's own extraction usually already names
    its room directly, so room_hint mainly covers a trailing fragment clause
    ("and a tv") that doesn't restate it. `room_id` is the caller's own
    already-grounded room (from task.connection — no more active_room_id
    fallback, removed system-wide); it's the fallback below when neither
    extraction nor room_hint names a room this turn."""
    known = {**(await known_fields(project_id, room_id)), **(known_override or {})}
    anchors = await _build_anchors(project_id)

    entities_result, relationships_result = await asyncio.gather(
        extract_entities(message, known), extract_relationships(message, anchors, [], []), return_exceptions=True
    )

    if isinstance(entities_result, Exception):
        logger.warning("resolve_context: extract_entities failed for project %s (%s: %s) — skipping structured extraction this turn", project_id, type(entities_result).__name__, entities_result)
        extracted = None
    else:
        extracted = entities_result

    if isinstance(relationships_result, Exception):
        logger.warning("resolve_context: extract_relationships failed for project %s (%s: %s) — skipping freeform extraction this turn", project_id, type(relationships_result).__name__, relationships_result)
        graph_extraction = None
    else:
        graph_extraction, _status = relationships_result

    room_resolution = RoomResolution(room_id=room_id)
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

    async def _resolve_and_register_room(room_type: Optional[str]) -> Optional[str]:
        """_resolve_room is a pure lookup now (see its docstring) — this
        folds a newly-resolved room's RoomType leaf into `proposed` (the
        same batch every other structured field goes through) instead of
        _resolve_room writing it immediately itself, which is what makes
        this whole function safe to call from a read-only resolve phase."""
        resolution = await _resolve_room(project_id, room_type)
        if resolution.new_room_type is not None:
            _add("roomType", f"Project.Rooms.{resolution.room_id}.RoomType", "RoomType", resolution.new_room_type, resolution.room_id)
        return resolution.room_id

    if extracted is not None:
        room_resolution = await _resolve_room(project_id, extracted.roomType or room_hint, room_id)
        room_id = room_resolution.room_id
        if room_resolution.new_room_type is not None:
            _add("roomType", f"Project.Rooms.{room_id}.RoomType", "RoomType", room_resolution.new_room_type, room_id)

        has_room_scoped_fields = extracted.materials or any(
            v is not None for v in (extracted.budgetOrRequirement, extracted.style, extracted.squareFootage, extracted.existingFurniture)
        )
        if room_id is None and has_room_scoped_fields:
            # Room-scoped facts with no explicit roomType and no room
            # already grounded for this turn (e.g. "make it modern" as the
            # very first message, or any message with no room mention — no
            # more active_room_id "continue the last room" fallback,
            # removed system-wide) still need SOME room to belong to — same
            # as the pre-cutover PartialContext.resolve_room(None) behavior:
            # create an unlabeled room rather than silently dropping the
            # data. No RoomType value to register here (there's no name to
            # write) — the room's own container is created lazily by
            # apply_to_graph's ensure_path when the first field below is
            # written.
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
            extra_room_id = await _resolve_and_register_room(extra.roomType)
            _add("budgetOrRequirement", f"Project.Rooms.{extra_room_id}.Budget", "Budget", extra.budgetOrRequirement, extra_room_id)

        for room_name in extracted.mentionedAdditionalRooms or []:
            if room_name is not None:
                await _resolve_and_register_room(room_name)

    freeform_mentions: list[FreeformMention] = []
    freeform_relationships: list[dict] = []
    if graph_extraction is not None:
        freeform_relationships = [{"source": e.source, "target": e.target, "relation": e.relation} for e in graph_extraction.new_edges]

        for new_node in graph_extraction.new_nodes:
            if new_node.type == "room":
                continue
            if _is_duplicate_of_structured_value(new_node.label, structured_values):
                continue
            hint = _OLD_TYPE_TO_HINT.get(new_node.type)
            preview = await map_to_canonical(new_node.label, hint, project_id, room_id, commit=False)
            freeform_mentions.append(FreeformMention(raw_entity=new_node.label, node_type_hint=hint, preview=preview))

        if graph_extraction.retracted_node_ids:
            logger.warning(
                "resolve_context: %d retraction(s) requested for project %s but not applied — "
                "KnowledgeNode has no active/superseded/retracted lifecycle yet, see ontology/PHASE7_CONTEXT_BUILDER.md",
                len(graph_extraction.retracted_node_ids), project_id,
            )

    return ResolvedBuild(
        project_id=project_id, room_id=room_id, room_resolution=room_resolution, proposed=proposed,
        freeform_mentions=freeform_mentions, freeform_relationships=freeform_relationships,
    )


async def commit_context(resolved: ResolvedBuild) -> BuildResult:
    """The write half of build_context. Freeform mentions are re-resolved
    for real here (not just committed off resolved.preview verbatim) — see
    FreeformMention's docstring for why: project state (and therefore the
    right existing-vs-new decision) can have changed since resolve_context
    ran, especially once turns start resolving several operations
    concurrently before committing any of them. Every proposed write is
    applied directly now — there is no confirmation gate for a critical-tier
    value that already differs from what's stored (see PENDING_GAP_ANALYSIS.md);
    a restatement simply overwrites, same as any other field."""
    written = await apply_to_graph(resolved.project_id, resolved.proposed)

    for mention in resolved.freeform_mentions:
        match = await map_to_canonical(mention.raw_entity, mention.node_type_hint, resolved.project_id, resolved.room_id, commit=True)
        written.append(match.canonical_path)

    return BuildResult(
        project_id=resolved.project_id, room_id=resolved.room_id, written=written,
        freeform_relationships=resolved.freeform_relationships,
    )


async def build_context(
    message: str,
    project_id: str,
    room_id: Optional[str] = None,
    known_override: Optional[dict] = None,
    room_hint: Optional[str] = None,
) -> BuildResult:
    resolved = await resolve_context(message, project_id, room_id, known_override, room_hint)
    return await commit_context(resolved)
