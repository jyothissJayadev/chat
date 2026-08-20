"""The single-pipeline turn orchestrator — replaces app.execution's
resolve/cluster/ordered-commit machinery and the separate build_context_node/
delete_context_node/retrieve_context_node/query_catalog_node/
generate_answer_node/generate_question_node/validate_completeness_node/
complete_project_node/handle_split_intents_node nodes in app/graph.py (see
the pipeline redesign plan). One node — app.graph.run_pipeline_node — wraps
run_pipeline() below; everything else here is a plain async function, not a
LangGraph node, since the branch structure (which of write/retrieval/query/
answer apply this turn) is entirely data-dependent per turn and doesn't map
cleanly onto static graph edges.

Dependency shape for one turn:
  - CONTEXT_UPDATE + CONTEXT_DELETE tasks ("first action") commit via ONE
    combined LLM call (app.llm.resolve_context_changes) — see
    _run_first_action. Deletion targets are picked by that call directly off
    the live tree text (replacing the old embedding-similarity
    canonical_mapper.resolve_deletion_target), and freeform entity
    reuse-vs-create still goes through canonical_mapper.map_to_canonical's
    algorithmic alias/embedding dedup.
  - CONTEXT_RETRIEVAL no longer runs its own node fetch. `pieces["project_context"]`
    (below) is now the complete, single source of truth for "what's in the
    project" — a CONTEXT_RETRIEVAL task only contributes its `.target` text,
    joined into `pieces["context_retrieval_request"]`, so generate_turn_reply
    can build context_summary itself straight off the tree already in its
    system prompt.
  - DATABASE_RETRIEVAL starts immediately, fully concurrent with everything
    else. ANSWER tasks produce no LLM call of their own any more — their
    `.target` text is just joined into `pieces["direct_answer_request"]` for
    the join step below.
  - project_context (app.context_builder.render_project_tree_text — the
    whole live tree, not just the active room) is fetched only after first
    action (if any) lands, same as pending_gap: it now grounds BOTH the
    ANSWER block and CONTEXT_RETRIEVAL's context_summary, so — unlike the
    old per-room known_fields lookup, which only ever grounded ANSWER and
    could safely start pre-write — it must see this turn's own writes before
    a "what's in my project now" question can be answered correctly.
  - Once first action (if any) + database query (if any) + project_context +
    pending_gap are all done, ONE join LLM call (app.llm.generate_turn_reply)
    runs — every turn, unconditionally — and turns whatever pieces actually
    happened (a direct answer, database results, changes made, a context
    request, the next open field) into the turn's single `reply`. This is
    the only LLM call that produces the turn's user-facing text; there is
    never a second one.
"""

import asyncio
import logging
import re
import time
from typing import Any, Optional
from uuid import uuid4

from langgraph.config import get_stream_writer
from pydantic import BaseModel
from rapidfuzz import fuzz

from app import calculation, canonical_mapper, context_builder, graph_store, inference, llm, question_engine, rag, versioning
from app.config import settings
from app.context_builder import ProposedWrite
from app.models import FIELD_TIERS, ProjectContext, TraceEntry
from app.tasks import TaskSpec, TaskType
from app.understanding import TASK_TYPE_TO_INTENT

logger = logging.getLogger(__name__)

_WRITE_TASK_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT)
_ROOM_CONTAINER_PATH_RE = re.compile(r"^Project\.Rooms\.[^.]+$")
# Same confidence bar app.graph._resolve_room_hint/room_type_matches already
# use for "is this the same room" — kept consistent rather than picking a
# new number.
_ROOM_NAME_MENTION_THRESHOLD = 82


def _stream_writer():
    """get_stream_writer() raises RuntimeError outside a real LangGraph run
    (e.g. a test calling run_pipeline() directly) — fall back to a no-op
    instead of making that a hard requirement."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _event: None


def _trace(
    node_name: str,
    model_used: Optional[str],
    input_summary: str,
    output_summary: str,
    start: float,
    *,
    llm_input: Optional[list[dict]] = None,
    llm_output: Optional[str] = None,
) -> TraceEntry:
    return TraceEntry(
        node_name=node_name,
        model_used=model_used,
        input_summary=input_summary[:200],
        output_summary=output_summary[:200],
        duration_ms=int((time.perf_counter() - start) * 1000),
        llm_input=llm_input,
        llm_output=llm_output,
    )


# ---------------------------------------------------------------------------
# First action: combined CONTEXT_UPDATE + CONTEXT_DELETE
# ---------------------------------------------------------------------------


class FirstActionResult(BaseModel):
    written: list[str] = []
    retracted: list[str] = []
    # {"path", "before", "after", "action"} per touched node — the raw
    # material app.llm.generate_turn_reply's changes_summary is built from.
    changes: list[dict] = []
    active_room_id: Optional[str] = None


def _fields_to_proposed_writes(fields: "llm.ContextChangeFields", room_id: Optional[str]) -> list[ProposedWrite]:
    proposed: list[ProposedWrite] = []

    def add(field_name: str, path: str, node_type: str, value) -> None:
        if value is None:
            return
        proposed.append(
            ProposedWrite(canonical_path=path, node_type=node_type, value=value, room_id=room_id, tier=FIELD_TIERS.get(field_name, "moderate"))
        )

    add("projectType", "Project.BasicInformation.ProjectType", "ProjectType", fields.projectType)
    add("overallBudget", "Project.Budget.Total", "Total", fields.overallBudget)
    add("timeline", "Project.Timeline.Value", "Value", fields.timeline)

    if room_id:
        add("roomType", f"Project.Rooms.{room_id}.RoomType", "RoomType", fields.roomType)
        add("budgetOrRequirement", f"Project.Rooms.{room_id}.Budget", "Budget", fields.budgetOrRequirement)
        add("style", f"Project.Rooms.{room_id}.Style", "Style", fields.style)
        add("squareFootage", f"Project.Rooms.{room_id}.SquareFootage", "SquareFootage", fields.squareFootage)
        add("existingFurniture", f"Project.Rooms.{room_id}.ExistingFurniture", "ExistingFurniture", fields.existingFurniture)
        for material in fields.materials or []:
            item_path = f"Project.Rooms.{room_id}.Materials.{canonical_mapper.slugify(material.item)}"
            add("materials", f"{item_path}.Label", "Label", material.item)
            add("materials", f"{item_path}.Material", "Material", material.material)
            if material.specification is not None:
                add("materials", f"{item_path}.Specification", "Specification", material.specification)

    return proposed


async def _apply_deletions(project_id: str, root_relative_paths: list[str], changes: list[dict], retracted: list[str]) -> None:
    """`root_relative_paths` is `result.deletion_targets` — root-relative,
    same convention as `connection` (see canonical_mapper.to_full_path).
    Converted to the "Project."-prefixed form once per target before ever
    touching graph_store, which stores/looks up the full path."""
    for root_relative_path in root_relative_paths:
        path = canonical_mapper.to_full_path(root_relative_path)
        node = await graph_store.find_one(project_id, path)
        if node is None or node.lifecycle == "retracted":
            continue
        before = node.value
        if _ROOM_CONTAINER_PATH_RE.match(path):
            ids = await versioning.retract_subtree(node)
            retracted.extend(ids)
            changes.append({"path": path, "before": before, "after": None, "action": "deleted_room"})
        else:
            await versioning.retract_node(node)
            retracted.append(node.node_id)
            changes.append({"path": path, "before": before, "after": None, "action": "deleted"})


async def _apply_entity_edit(project_id: str, mention: "llm.FreeformEntityChange", changes: list[dict], written: list[str]) -> None:
    """Writes `mention.value` directly onto the existing entity's
    `mention.field` leaf (e.g. Rooms.<id>.Furniture.sofa.Material) — the
    path/field pair app.llm._validate_context_changes already confirmed
    exist and are legal before this ever runs. Bypasses
    canonical_mapper.map_to_canonical entirely: this IS the entity (named by
    exact path, not re-derived by embedding similarity), so there's no
    reuse-vs-create decision left to make.

    `mention.existing_path` is root-relative (same convention as
    `connection` — see canonical_mapper.to_full_path); graph_store wants the
    "Project."-prefixed form, so every path used here is converted once up
    front rather than passing the root-relative string straight into
    graph_store.find_one, which can never match anything (this was the
    actual bug behind session a3a41e7a's "does not exist in CURRENT DATA
    TREE" failures on otherwise-correct edits)."""
    full_existing_path = canonical_mapper.to_full_path(mention.existing_path)
    entity = await graph_store.find_one(project_id, full_existing_path)
    if entity is None:
        # The validator confirmed this path existed when the LLM call ran;
        # a concurrent change mid-turn is the only way it's gone now — treat
        # as a no-op rather than crash on a stale reference.
        return
    leaf_path = f"{full_existing_path}.{mention.field}"
    before_node = await graph_store.find_one(project_id, leaf_path)
    before_value = before_node.value if before_node else None
    write = ProposedWrite(canonical_path=leaf_path, node_type=mention.field, value=mention.value, room_id=entity.room_id)
    just_written = await context_builder.apply_to_graph(project_id, [write])
    if just_written:
        after_node = await graph_store.find_one(project_id, leaf_path)
        changes.append(
            {
                "path": leaf_path,
                "before": before_value,
                "after": after_node.value if after_node else None,
                "action": "created" if before_value is None else "updated",
            }
        )
        written.append(leaf_path)


async def _write_leaf(
    project_id: str,
    canonical_path: str,
    node_type: str,
    value: Any,
    room_id: Optional[str],
    changes: list[dict],
    written: list[str],
) -> None:
    """One leaf write via context_builder.apply_to_graph, with a `changes`
    entry recorded the same shape as _apply_entity_edit / _apply_update's
    structured-field loop — factored out so Quantity/Material/Value leaves
    on composite entities (parts/properties) share one code path instead of
    each callsite re-implementing the before/after/apply/record dance. A
    no-value (None) write is a no-op, matching _fields_to_proposed_writes'
    `add` guard."""
    if value is None:
        return
    before_node = await graph_store.find_one(project_id, canonical_path)
    before_value = before_node.value if before_node else None
    write = ProposedWrite(canonical_path=canonical_path, node_type=node_type, value=value, room_id=room_id)
    just_written = await context_builder.apply_to_graph(project_id, [write])
    if just_written:
        after_node = await graph_store.find_one(project_id, canonical_path)
        changes.append(
            {
                "path": canonical_path,
                "before": before_value,
                "after": after_node.value if after_node else None,
                "action": "created" if before_value is None else "updated",
            }
        )
        written.append(canonical_path)


async def _apply_properties(
    parent_instance_path: str,
    properties: list["llm.PropertyChange"],
    project_id: str,
    room_id: Optional[str],
    changes: list[dict],
    written: list[str],
) -> None:
    """Writes each property as a Properties.<slug> child of
    `parent_instance_path`: map_property_to_canonical resolves/creates the
    instance + its Label (=name), then this writes the Value leaf (=value)
    — updating an existing property's value in place on alias reuse rather
    than creating a duplicate. `parent_instance_path` is the full
    canonical path (with "Project." prefix) of the entity or part these
    properties attach to."""
    for prop in properties:
        match = await canonical_mapper.map_property_to_canonical(prop.name, prop.value, parent_instance_path, project_id, room_id, commit=True)
        written.append(match.canonical_path)
        if match.created_new:
            changes.append({"path": f"{match.canonical_path}.Label", "before": None, "after": prop.name, "action": "created"})
        await _write_leaf(project_id, f"{match.canonical_path}.Value", "Value", prop.value, room_id, changes, written)


async def _apply_parts(
    parent_instance_path: str,
    parts: list["llm.PartChange"],
    project_id: str,
    room_id: Optional[str],
    changes: list[dict],
    written: list[str],
) -> None:
    """Writes each part as a Parts.<slug> child of `parent_instance_path`
    via canonical_mapper.map_part_to_canonical, then the part's own
    Material/Quantity leaves and Properties children. One level only —
    PartChange has no `parts` field, so there is no recursion here. A part
    with existing_path is an edit of an existing part's leaf (handled by
    the existing_path branch); otherwise it's a new part nested under the
    parent instance."""
    for part in parts:
        if part.existing_path is not None and part.field is not None:
            # Editing an existing part's leaf — same path/field contract as
            # _apply_entity_edit, just on a part.
            await _write_leaf(
                project_id, f"{canonical_mapper.to_full_path(part.existing_path)}.{part.field}", part.field, part.value,
                room_id, changes, written,
            )
            continue
        match = await canonical_mapper.map_part_to_canonical(part.raw_entity, parent_instance_path, project_id, room_id, commit=True)
        written.append(match.canonical_path)
        if match.created_new:
            changes.append({"path": f"{match.canonical_path}.Label", "before": None, "after": part.raw_entity, "action": "created"})
        await _write_leaf(project_id, f"{match.canonical_path}.Material", "Material", part.material, room_id, changes, written)
        await _write_leaf(project_id, f"{match.canonical_path}.Quantity", "Quantity", part.quantity, room_id, changes, written)
        await _apply_properties(match.canonical_path, part.properties, project_id, room_id, changes, written)


def _is_room_name_mention(raw_entity: str, room_hint: Optional[str]) -> bool:
    """True when `raw_entity` is (or closely paraphrases) `room_hint` itself
    — a defensive backstop for app.llm.resolve_context_changes' rule-5
    exception (see app.prompts.RESOLVE_CONTEXT_CHANGES_SYSTEM_TEMPLATE):
    even with that exception, an LLM slip on a NEW-room operation could
    still emit a freeform_entities mention whose raw_entity is just the
    room's own name (e.g. "Add living room" -> a spurious Materials
    instance labeled "living room") — this catches that before it ever
    reaches canonical_mapper.map_to_canonical, rather than relying on the
    prompt alone. `room_hint` is only non-None for a genuinely NEW room
    (see app.canonical_mapper.split_connection); an existing room's mentions
    are never filtered by this, since there's no such ambiguity there."""
    if not room_hint:
        return False
    return fuzz.ratio(raw_entity.strip().lower(), room_hint.strip().lower()) >= _ROOM_NAME_MENTION_THRESHOLD


async def _apply_update(
    project_id: str,
    room_id: Optional[str],
    result: "llm.ContextChangeResult",
    proposed_extra: list[ProposedWrite],
    changes: list[dict],
    written: list[str],
    room_hint: Optional[str] = None,
    parent_instance_path: Optional[str] = None,
) -> None:
    """`parent_instance_path` is the full canonical path (with "Project."
    prefix) of an existing freeform instance the operation's `connection`
    pointed DEEPER than Rooms.<id> at (e.g. "Project.Rooms.<id>.Furniture.beds"
    — the bed a bedcover is being added TO). When set, parts emitted on a
    mention attach under that parent instance via map_part_to_canonical;
    when None (room-level connection), parts attach under the mention's own
    just-resolved instance (the "add a sofa with a cover" composite-new case).
    Properties always attach to the mention's own instance.

    The structured-fields block (rules 2) writes first, then each
    freeform_entities mention. A mention with existing_path is an edit of
    an existing entity's leaf (rule 3) — but it may ALSO carry parts/
    properties/quantity (e.g. editing the bed by adding a bedcover part to
    it), so those are still processed after the leaf edit instead of the
    old `continue` short-circuit that silently dropped them."""
    proposed = proposed_extra + _fields_to_proposed_writes(result.fields, room_id)

    before_map: dict[str, object] = {}
    for write in proposed:
        existing = await graph_store.find_one(project_id, write.canonical_path)
        before_map[write.canonical_path] = existing.value if existing else None

    just_written = await context_builder.apply_to_graph(project_id, proposed)
    for path in just_written:
        after_node = await graph_store.find_one(project_id, path)
        changes.append(
            {
                "path": path,
                "before": before_map.get(path),
                "after": after_node.value if after_node else None,
                "action": "created" if before_map.get(path) is None else "updated",
            }
        )
        written.append(path)

    for mention in result.freeform_entities:
        if _is_room_name_mention(mention.raw_entity, room_hint):
            continue
        # Resolve the mention's own entity instance first — its path is the
        # anchor for any quantity/properties on it, and the fallback anchor
        # for parts when the connection wasn't deeper than Rooms.<id>.
        if mention.existing_path is not None:
            if mention.field is not None:
                await _apply_entity_edit(project_id, mention, changes, written)
            entity_path = f"Project.{mention.existing_path}"
        else:
            match = await canonical_mapper.map_to_canonical(mention.raw_entity, mention.node_type_hint, project_id, room_id, commit=True)
            written.append(match.canonical_path)
            if match.created_new:
                changes.append({"path": f"{match.canonical_path}.Label", "before": None, "after": mention.raw_entity, "action": "created"})
            entity_path = match.canonical_path

        # Quantity on the entity itself ("two pillows" -> Quantity="2").
        await _write_leaf(project_id, f"{entity_path}.Quantity", "Quantity", mention.quantity, room_id, changes, written)
        # Named properties (color/finish/fabric) on the entity.
        await _apply_properties(entity_path, mention.properties, project_id, room_id, changes, written)
        # Parts: attach to the connection's parent instance when the
        # operation targeted an existing instance deeper than the room
        # ("add a bedcover to the bed"), else to the mention's own instance
        # ("add a sofa with a cover" — sofa is new this turn).
        parts_parent = parent_instance_path or entity_path
        await _apply_parts(parts_parent, mention.parts, project_id, room_id, changes, written)


async def _run_first_action(project_id: str, write_tasks: list[TaskSpec], start: float) -> tuple[FirstActionResult, TraceEntry]:
    tree_text = await context_builder.render_project_tree_text(project_id)
    # One snapshot of the live graph, fetched once and handed to
    # llm.resolve_context_changes as ground truth for its semantic
    # validation layer (does a claimed deletion_target/existing_path really
    # exist? is a claimed field legal for that node's own type?) — see
    # app.llm._validate_context_changes. Keyed root-relative (canonical_
    # mapper.to_root_relative), NOT graph_store's raw "Project."-prefixed
    # canonical_path: the resolver's existing_path/deletion_targets are
    # root-relative, same convention as connection and as CURRENT DATA TREE
    # itself (context_builder.render_project_tree_text) — comparing them
    # against full paths meant a correctly-instructed model's output could
    # never validate as "exists," silently dropping every genuine edit/
    # deletion whose path didn't happen to already carry a stray "Project."
    # prefix. See the postmortem on session a3a41e7a.
    live_nodes = await graph_store.find_nodes(project_id, lifecycle="active")
    existing_paths = {canonical_mapper.to_root_relative(n.canonical_path) for n in live_nodes}
    node_types_by_path = {canonical_mapper.to_root_relative(n.canonical_path): n.node_type for n in live_nodes}

    operations = [{"text": t.target or "", "intent": TASK_TYPE_TO_INTENT[t.type], "connection": t.connection} for t in write_tasks]
    capture: dict = {}
    results, issues = await llm.resolve_context_changes(tree_text, operations, existing_paths, node_types_by_path, capture=capture)

    # Two ops in the SAME turn naming the same not-yet-existing room (e.g.
    # "add a kids bedroom" + "put a bunk bed in the kids bedroom") must
    # create that room once, not twice — this dict is what used to be
    # app.execution._cluster's job, now just a plain local dedup since every
    # write op in a turn commits sequentially in one pass instead of
    # independently-resolved concurrent branches.
    new_rooms: dict[str, str] = {}
    changes: list[dict] = []
    written: list[str] = []
    retracted: list[str] = []
    active_room_id: Optional[str] = None

    for task, result in zip(write_tasks, results):
        room_id, room_hint, parent_instance_path = canonical_mapper.split_connection(task.connection)
        room_write: list[ProposedWrite] = []
        if room_id is None and room_hint:
            key = room_hint.strip().lower()
            room_id = new_rooms.get(key)
            if room_id is None:
                room_id = uuid4().hex[:8]
                new_rooms[key] = room_id
                room_write.append(
                    ProposedWrite(
                        canonical_path=f"Project.Rooms.{room_id}.RoomType", node_type="RoomType", value=room_hint,
                        room_id=room_id, tier=FIELD_TIERS.get("roomType", "critical"),
                    )
                )
        if room_id:
            active_room_id = room_id

        if result.intent == "CONTEXT_DELETE":
            await _apply_deletions(project_id, result.deletion_targets, changes, retracted)
            continue

        await _apply_update(
            project_id, room_id, result, room_write, changes, written,
            room_hint=room_hint, parent_instance_path=parent_instance_path,
        )

    # Anything the LLM claimed but couldn't back up after a validation
    # retry (app.llm.resolve_context_changes) — surfaced as a "failed"
    # change so generate_turn_reply's changes_summary can mention it,
    # rather than silently doing nothing for that one claim.
    for issue in issues:
        changes.append({"path": None, "before": None, "after": None, "action": "failed", "reason": issue.detail})

    result = FirstActionResult(written=written, retracted=retracted, changes=changes, active_room_id=active_room_id)
    entry = _trace(
        "first_action",
        settings.model_extraction,
        "; ".join(t.target or "" for t in write_tasks),
        f"wrote {len(result.written)}, retracted {len(result.retracted)}",
        start,
        llm_input=capture.get("messages"),
        llm_output=capture.get("raw_output"),
    )
    return result, entry


# ---------------------------------------------------------------------------
# CONTEXT_RETRIEVAL
#
# No dedicated node fetch any more — pieces["project_context"] (the full,
# post-write project tree, built by run_pipeline below) already contains
# everything a scoped fetch would have returned, so a CONTEXT_RETRIEVAL task
# only needs to contribute the client's own request text; generate_turn_reply
# builds context_summary itself from that text + the tree already in its
# system prompt. See app.retrieval for the older per-node-fetch approach
# (still exercised by tests/test_retrieval.py, no longer called live).
# ---------------------------------------------------------------------------
# DATABASE_RETRIEVAL
# ---------------------------------------------------------------------------


async def _run_database_query(query_tasks: list[TaskSpec], message_fallback: str) -> tuple[list[dict], TraceEntry]:
    start = time.perf_counter()
    results: list[dict] = []
    capture: dict = {}
    llm_input: Optional[list[dict]] = None
    llm_output: Optional[str] = None
    query_text = ""
    for task in query_tasks:
        query_text = task.target or message_fallback
        keywords = await llm.generate_search_keywords(query_text, capture=capture)
        llm_input = capture.get("messages")
        llm_output = capture.get("raw_output")
        items = await rag.query_catalog(keywords or query_text)
        results.extend({"title": item.title, "description": item.description} for item in items)
    entry = _trace(
        "database_query",
        settings.model_question_gen,
        query_text,
        f"{len(results)} catalog result(s)",
        start,
        llm_input=llm_input,
        llm_output=llm_output,
    )
    return results, entry


# ---------------------------------------------------------------------------
# Project completion — moved from the old complete_project_node
# ---------------------------------------------------------------------------


async def _build_assumptions(project_id: str) -> list[str]:
    inferred = await inference.list_inferred(project_id)
    calculated = await calculation.list_calculated(project_id)
    assumptions = []
    for node in inferred + calculated:
        field_name = context_builder.LEAF_TO_FIELD_NAME.get(node.node_type, node.node_type)
        assumptions.append(f"{field_name}: assumed {node.value}")
    return assumptions


async def _materialize_project(state: dict) -> None:
    project_id = state["project_id"]
    summary = await context_builder.materialize_project_summary(project_id)
    assumptions = await _build_assumptions(project_id)
    project = ProjectContext(session_id=state["session_id"], project_id=project_id, summary=summary, assumptions=assumptions)
    await project.insert()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(state: dict) -> dict:
    project_id = state["project_id"]
    tasks: list[TaskSpec] = state.get("tasks") or []
    write_tasks = [t for t in tasks if t.type in _WRITE_TASK_TYPES]
    retrieval_tasks = [t for t in tasks if t.type == TaskType.RETRIEVE_CONTEXT]
    query_tasks = [t for t in tasks if t.type == TaskType.DATABASE_QUERY]
    answer_tasks = [t for t in tasks if t.type == TaskType.ANSWER]
    direct_answer_request = " ".join(t.target for t in answer_tasks if t.target) or None
    context_retrieval_request = " ".join(t.target for t in retrieval_tasks if t.target) or None

    writer = _stream_writer()
    trace: list[TraceEntry] = []

    # DATABASE_RETRIEVAL starts immediately, independent of the write step —
    # it doesn't read the KnowledgeNode graph at all. project_context does
    # NOT start here any more: it's now the single, complete, post-write
    # project tree (context_builder.render_project_tree_text) that grounds
    # BOTH the ANSWER block and CONTEXT_RETRIEVAL's context_summary, so it
    # needs the same "wait for first action" guarantee pending_gap already
    # requires — starting it early would let it race ahead of this turn's
    # own writes and hand the join step stale (pre-write) data for a
    # question about what THIS turn just changed.
    query_future = asyncio.ensure_future(_run_database_query(query_tasks, state["message"])) if query_tasks else None

    first_action: Optional[FirstActionResult] = None
    if write_tasks:
        start = time.perf_counter()
        first_action, first_action_entry = await _run_first_action(project_id, write_tasks, start)
        writer({"type": "operation_progress", "stage": "first_action", "ok": True})
        trace.append(first_action_entry)

    if retrieval_tasks:
        writer({"type": "operation_progress", "stage": "context_retrieval", "ok": True})

    # project_context and pending_gap both read the live KnowledgeNode graph
    # and must only start now — after the write step above (if any) has
    # landed. Neither depends on the other's result, so they run
    # concurrently with each other from here.
    project_context_future = asyncio.ensure_future(context_builder.render_project_tree_text(project_id))
    # ProjectType is asked at most once: if the turn we're about to process
    # was itself answering a ProjectType question (this turn's incoming
    # pending_gap), it's used up regardless of whether the reply actually
    # filled it in — see ChatSession.project_type_skipped.
    incoming_gap_paths = {g["canonical_path"] for g in (state.get("pending_gap") or {}).get("gaps") or []}
    project_type_skipped = bool(state.get("project_type_skipped")) or (
        "Project.BasicInformation.ProjectType" in incoming_gap_paths
    )
    gap_future = asyncio.ensure_future(
        question_engine.find_knowledge_gaps(
            project_id, state.get("active_room_id"), state.get("skipped_rooms") or [], project_type_skipped
        )
    )

    query_results: list[dict] = []
    if query_future is not None:
        query_results, query_entry = await query_future
        writer({"type": "operation_progress", "stage": "database_query", "ok": True})
        trace.append(query_entry)

    project_context = await project_context_future
    gap_batch = await gap_future

    active_room_id = (first_action.active_room_id if first_action else None) or state.get("active_room_id")
    complete = gap_batch is None

    if complete:
        await _materialize_project(state)

    # generate_turn_reply runs unconditionally, exactly once per turn — it's
    # the only place a direct answer, database results, changes, context, or
    # the next question get turned into user-facing text, and there is never
    # a second LLM call competing with it for that job (see module docstring).
    # project_context (the full tree) appears exactly once here — it's
    # rendered into the system prompt only (see prompts.build_turn_reply_system
    # / format_project_context) and deliberately excluded from the JSON blob
    # prompts.turn_reply_user dumps for the rest of these pieces, so the
    # (potentially large) tree is never sent to the model twice.
    pieces = {
        "direct_answer_request": direct_answer_request,
        "project_context": project_context,
        "changes": first_action.changes if first_action else [],
        "context_retrieval_request": context_retrieval_request,
        "database_results": query_results,
        "pending_gap": [g.field_label for g in gap_batch.gaps] if gap_batch else None,
    }
    start = time.perf_counter()
    capture: dict = {}
    turn_reply = await llm.generate_turn_reply(pieces, capture=capture)
    trace.append(
        _trace(
            "generate_turn_reply",
            settings.model_question_gen,
            str(pieces)[:200],
            turn_reply.reply,
            start,
            llm_input=capture.get("messages"),
            llm_output=capture.get("raw_output"),
        )
    )
    writer({"type": "operation_progress", "stage": "summary", "ok": True})

    return {
        "reply": turn_reply.reply,
        "context_summary": turn_reply.context_summary,
        "changes_summary": turn_reply.changes_summary,
        "is_question": turn_reply.is_question,
        "pending_gap": gap_batch.model_dump() if gap_batch else None,
        "active_room_id": active_room_id,
        "project_type_skipped": project_type_skipped,
        "complete": complete,
        "trace": trace,
    }
