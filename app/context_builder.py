"""Context tree read/write primitives shared by app/pipeline.py.

The old per-message extraction pipeline that used to live here
(extract_entities/extract_relationships/resolve_context/commit_context/
build_context, plus their RoomResolution/FreeformMention/ResolvedBuild/
BuildResult support types) is gone — see the pipeline redesign plan.
Structured-field extraction and freeform-entity/deletion resolution now
happen in ONE combined call per turn (app.llm.resolve_context_changes),
covering every CONTEXT_UPDATE/CONTEXT_DELETE task together, orchestrated by
app.pipeline._run_first_action. What remains here are the primitives that
call still needs: writing a resolved ProposedWrite to its canonical leaf
(apply_to_graph), reading back known values (known_fields,
materialize_project_summary), and rendering the live tree as text for the
classifier/resolver prompts (render_project_tree_text)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel

from app import canonical_mapper, graph_store
from app.models import ChangedBy, KnowledgeNode, utcnow
from app.versioning import record_version

Tier = Literal["critical", "moderate", "optional"]

# Leaf node_type -> the FIELD_TIERS key that governs its conflict tier.
# Materials leaves (Label/Material/Specification) aren't listed individually
# here — FIELD_TIERS itself only has one "materials" entry covering the
# whole list, applied uniformly by app.pipeline. Public (not `_`-prefixed):
# also used by app/pipeline.py and app/question_engine.py for field labels,
# confirmation prompts, and assumption summaries.
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


class ProposedWrite(BaseModel):
    canonical_path: str
    node_type: str
    value: Any
    room_id: Optional[str] = None
    tier: Tier = "moderate"
    # Everything app.pipeline proposes today comes from the client's own
    # message (structured extraction or freeform mapping) — see
    # ontology/PHASE8_VERSIONING.md for why "inferred"/"system_default"
    # aren't produced by any call site yet.
    changed_by: ChangedBy = "user_message"


async def known_fields(project_id: str, room_id: Optional[str]) -> dict[str, Any]:
    """Project- and room-scoped known values, in the same shape
    generate_answer's/generate_turn_summary's prompts already expect. Public
    (not `_`-prefixed): app/pipeline.py reuses this directly for every step
    that needs to show "what's known so far" to a model, rather than each
    re-deriving it."""
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
    "rooms": [{"roomType": ..., "materials": [...]}, ...]}. Used by
    app.pipeline._materialize_project to populate ProjectContext.summary."""
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
# uniformly across all freeform node types rather than keyed per-type:
# app.canonical_mapper._create_instance today only ever writes the Label
# leaf (Material/Specification/Notes/Quantity aren't wired up by any write
# path yet beyond the composite-entity pipeline's _write_leaf), so this
# mostly renders "Label=..." alone — the other keys are included only if
# some writer starts populating them.
_FREEFORM_LEAF_KEYS = ("Label", "Material", "Specification", "Quantity", "Notes", "Value")
# Composite-entity child containers (see ontology/v1.yaml Parts/Properties):
# rendered as nested sub-branches under their parent freeform instance, NOT
# as top-level room children (the room-level instance scan below only picks
# _ROOM_FREEFORM_TYPES, so Parts/Properties never surface there — they're
# discovered here by walking each instance's own subtree).
_COMPOSITE_CHILD_TYPES = ("Parts", "Properties")
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
    container the SAME node_type as the instances it holds, so a
    container's last path segment is always its own node_type — a real
    instance's slug never is."""
    return node.canonical_path.rsplit(".", 1)[-1] == node.node_type


def _freeform_instance_line(instance: KnowledgeNode, by_path: dict[str, KnowledgeNode]) -> tuple[str, list[str]]:
    """Renders a freeform instance (Furniture/Materials/Attributes/Parts/
    Properties/Unmapped) as a tree node: the instance's own leaf values on
    the first detail line, then any composite children (Parts/Properties
    instances nested under it) as recursive sub-branches. Recursion bottoms
    out at Properties (terminal — no children of its own per ontology/v1.yaml)
    and at Parts whose own sub-details folded into properties/quantity (one
    level only). Returns the (header, detail_lines) shape _tree_branch
    composes by prefixing detail with the parent's continuation string."""
    slug = instance.canonical_path.rsplit(".", 1)[-1]
    header = f"{instance.node_type}.{slug}"
    detail: list[str] = []
    leaf_parts = []
    for key in _FREEFORM_LEAF_KEYS:
        leaf = by_path.get(f"{instance.canonical_path}.{key}")
        if leaf is not None and leaf.value is not None:
            leaf_parts.append(f'{key}="{leaf.value}"')
    if leaf_parts:
        detail.append(", ".join(leaf_parts))
    # Composite child instances (Parts/Properties) grouped under their
    # container — discovered by path prefix + node_type rather than a
    # separate graph query, since by_path already holds the whole active
    # tree for this render call.
    child_items: list[tuple[str, list[str]]] = []
    for child_type in _COMPOSITE_CHILD_TYPES:
        container_path = f"{instance.canonical_path}.{child_type}"
        child_instances = sorted(
            (n for n in by_path.values()
             if n.node_type == child_type
             and n.canonical_path.startswith(f"{container_path}.")
             and not _is_container(n)),
            key=lambda n: n.canonical_path,
        )
        if child_instances:
            rendered = [_freeform_instance_line(n, by_path) for n in child_instances]
            child_items.append((child_type, _tree_branch(rendered)))
    if child_items:
        detail.extend(_tree_branch(child_items))
    return header, detail


async def render_project_tree_text(project_id: str) -> str:
    """Plain-text, root-relative rendering of the live KnowledgeNode tree —
    fed into classify_operations'/resolve_context_changes' system prompts
    (see app.prompts) so the model can ground each operation's `connection`/
    deletion targets against the project's actual current state.

    Root-relative: no "Project." prefix on any path, matching the convention
    app.llm.Operation.connection itself uses (e.g. "Rooms.a1b2c3d4",
    not "Project.Rooms.a1b2c3d4"). Value-only: no node_id/version/confidence/
    status/lifecycle/tenant_id/embedding ever appears, only canonical
    structure + value — and a null/absent field is omitted entirely rather
    than shown as a blank, so a half-filled project reads as exactly what's
    known so far."""
    nodes = await graph_store.find_nodes(project_id, lifecycle="active")
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
    return "\n".join(["Project", *_tree_branch(sections)])


async def apply_to_graph(project_id: str, writes: list[ProposedWrite]) -> list[str]:
    """Writes each ProposedWrite to its canonical leaf, creating any missing
    ancestor container along the way (canonical_mapper.ensure_path). A new
    leaf, or an existing one whose value actually changes, gets its `version`
    bumped (or set to 1) AND an append-only KnowledgeNodeVersion row via
    app.versioning.record_version — see ontology/PHASE8_VERSIONING.md, and is
    included in the returned list. A write whose value is identical to
    what's already stored is a true no-op: no version bump, no history row,
    and NOT included in the returned list. Critical-tier fields overwrite
    the same as any other field — there is no confirmation gate for a value
    that already differs from what's stored; app.pipeline's changes_summary
    covers transparency instead."""
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
