"""Task dispatcher — see ARCHITECTURE_BASELINE.md Phase 1/2/11. understanding.py
decides WHAT a turn needs to do (pure, no side effects); this module is the
only place allowed to call functions that mutate GraphState to make it
happen. Wired into build_graph() as of Phase 2 — see
app.graph.handle_split_intents_node, which now just delegates here.

Generalizes what used to be handle_split_intents_node's own inline concurrent
dispatch-and-merge (asyncio.gather over independent branches, then a single
merged state update) to run over a list[TaskSpec] instead of a list of raw
intent strings — same shape, different source of truth, per the plan.

ANSWER is deliberately not in the dispatch table: generate_answer_node reads
KnowledgeNode state that build_context_node may just have written, so it
must run after any EDIT_CONTEXT task in this batch has completed, not
concurrently with it — same ordering the old split-intent path enforced via
needs_answer + _route_post_completeness. execute() sets needs_answer
whenever EDIT_CONTEXT is part of the batch (not whenever ANSWER is) —
that's the actual signal app.graph._route_after_split needs: a batch with no
EDIT_CONTEXT task has nothing to validate/complete, so it routes straight to
generate_answer regardless of this flag; a batch WITH one must resolve
completeness first, then still come back and answer whatever else was in
the batch (an ANSWER, a DATABASE_QUERY, or a RETRIEVE_CONTEXT —
generate_answer_node doesn't care which, it just answers state["message"]
using whatever got merged into state["retrieved"])."""

import asyncio
import logging
import time

from langgraph.config import get_stream_writer

from app import canonical_mapper, context_builder, llm
from app.canonical_mapper import slugify
from app.models import TraceEntry
from app.tasks import TaskSpec, TaskType

logger = logging.getLogger(__name__)


_CONTEXT_TASK_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT)


def _stream_writer():
    """get_stream_writer() raises RuntimeError when called outside a real
    LangGraph run (e.g. tests/test_execution.py calling execute() directly,
    or any future non-graph caller) — this falls back to a no-op instead of
    making that a hard requirement just to support optional progress events."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _event: None


def _progress_event(task: TaskSpec, state: dict, ok: bool) -> dict:
    return {
        "type": "operation_progress",
        "task_type": task.type.value,
        "target": (task.target or state.get("message", ""))[:200],
        "ok": ok,
    }

# A ResolvedTask is (task, target_keys, resolved) — resolved is a
# context_builder.ResolvedBuild for an EDIT_CONTEXT task, always None for a
# DELETE_CONTEXT task (see _resolve_write_task's docstring for why delete
# doesn't need one).


async def _resolve_write_task(task: TaskSpec, state: dict) -> tuple[TaskSpec, set[str], "context_builder.ResolvedBuild | None"]:
    """The parallel "resolve" half of write-clustering (see the
    classifier-redesign plan: resolve every EDIT_CONTEXT/DELETE_CONTEXT task
    concurrently, THEN group by overlapping target before committing
    anything). Returns the task's connectivity key set — canonical paths its
    writes would touch, plus a room-level key and a not-yet-created-room-name
    key so two tasks racing to create the SAME new room/node still cluster
    together even though their own resolved paths don't literally match yet
    (see ResolvedBuild.room_resolution's docstring).

    Only EDIT_CONTEXT gets a real ResolvedBuild back — DELETE_CONTEXT's own
    resolve (app.graph.resolve_delete_target) is two read-only lookups, cheap
    enough that delete_context_node just re-runs them itself at commit time
    rather than this function threading a resolution through; only its
    target path (for clustering) is needed here."""
    from app.graph import resolve_delete_target

    message = task.target or state["message"]

    # A pre-resolve clustering key straight off the classifier's own
    # connection — doesn't wait for (or depend on agreeing with)
    # resolve_context's own extraction, so two ops the classifier already
    # grounded to the same path cluster together even if their independent
    # extractions would otherwise disagree. See canonical_mapper.
    # is_grounded_connection.
    pre_keys: set[str] = set()
    if task.connection and canonical_mapper.is_grounded_connection(task.connection) and task.connection != "Project":
        pre_keys.add(f"Project.{task.connection}")

    if task.type == TaskType.EDIT_CONTEXT:
        # connection is an anchor hint, not a final write path (see the
        # classifier-connection plan): a grounded room segment is the room
        # this task resolves against (no more active_room_id fallback —
        # removed system-wide), and a free-text connection (a
        # not-yet-existing room/entity name) overrides room_hint —
        # generalizing TaskSpec.room_hint's old regex-detected-vocabulary
        # source to the classifier's own project-grounded guess. Neither
        # ever bypasses resolve_context/canonical_mapper's own leaf-level
        # create-vs-update resolution.
        connection_room_id, connection_room_hint = canonical_mapper.split_connection(task.connection)
        resolved = await context_builder.resolve_context(
            message,
            state["project_id"],
            connection_room_id,
            room_hint=connection_room_hint or task.room_hint,
        )
        keys = pre_keys | {p.canonical_path for p in resolved.proposed}
        keys.update(m.preview.canonical_path for m in resolved.freeform_mentions)
        if resolved.room_id:
            keys.add(f"Project.Rooms.{resolved.room_id}")
        if resolved.room_resolution.new_room_type:
            keys.add(f"new-room:{slugify(resolved.room_resolution.new_room_type)}")
        return task, keys, resolved

    preview = await resolve_delete_target(state, task)
    keys = pre_keys
    if preview.canonical_path:
        keys.add(preview.canonical_path)
    if preview.room_id:
        keys.add(f"Project.Rooms.{preview.room_id}")
    return task, keys, None


def _cluster(
    resolved_tasks: list[tuple[TaskSpec, set[str], object]],
) -> list[list[tuple[TaskSpec, set[str], object]]]:
    """Union-find over shared target keys — two tasks land in the same
    cluster iff their resolved target-key sets intersect (same canonical
    path, same room, or same not-yet-created room/entity name — see
    _resolve_write_task). A task with an empty key set (nothing resolvable,
    e.g. a delete with no confident match) is its own singleton cluster and
    runs independently of everything else."""
    n = len(resolved_tasks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    key_to_index: dict[str, int] = {}
    for i, (_task, keys, _resolved) in enumerate(resolved_tasks):
        for key in keys:
            if key in key_to_index:
                union(i, key_to_index[key])
            else:
                key_to_index[key] = i

    clusters: dict[int, list] = {}
    for i, item in enumerate(resolved_tasks):
        clusters.setdefault(find(i), []).append(item)
    return list(clusters.values())


async def _commit_cluster(
    cluster: list[tuple[TaskSpec, set[str], object]], state: dict, writer
) -> list[tuple[TaskSpec, object]]:
    """Commits one cluster's tasks IN ORDER: every DELETE_CONTEXT task first
    (original relative order), then every EDIT_CONTEXT task (original
    relative order) — delete before update/create, per the classifier-
    redesign plan. Sequential, not gathered, within a cluster: each commit
    must see the previous ones' effects, which is what lets two tasks that
    both resolved against the same not-yet-created room/entity (that's WHY
    they're in this cluster) end up creating it once instead of racing to
    create it twice.

    Only the very first task committed in the cluster reuses the
    already-resolved plan from the parallel resolve pass in execute() — every
    task after that re-resolves fresh right before committing (build_context_node's
    resolved=None fallback; delete_context_node always re-resolves itself
    regardless), since a resolution computed before an earlier same-cluster
    write landed can no longer be trusted. Each task's own exception is
    caught and returned in place of its outcome (mirrors execute()'s old
    flat return_exceptions=True gather) so one failure doesn't stop the rest
    of the cluster's tasks from still being attempted.

    `writer` gets one `operation_progress` custom stream event per task as
    it finishes committing — since clusters commit sequentially, this is what
    lets a live client see each write land one at a time instead of only
    finding out the whole multi-operation batch finished, all at once, after
    the fact (see app.graph.handle_split_intents_node and app.chat.run_chat_turn)."""
    from app.graph import build_context_node, delete_context_node

    deletes = [item for item in cluster if item[0].type == TaskType.DELETE_CONTEXT]
    edits = [item for item in cluster if item[0].type == TaskType.EDIT_CONTEXT]
    ordered = deletes + edits

    results: list[tuple[TaskSpec, object]] = []
    for position, (task, _keys, resolved) in enumerate(ordered):
        try:
            if task.type == TaskType.DELETE_CONTEXT:
                outcome = await delete_context_node(state, task)
            else:
                outcome = await build_context_node(state, task, resolved=resolved if position == 0 else None)
        except Exception as exc:
            logger.warning(
                "execute: task %s failed (%s: %s) — other tasks in this batch still complete",
                task.type, type(exc).__name__, exc,
            )
            outcome = exc
        writer(_progress_event(task, state, ok=not isinstance(outcome, Exception)))
        results.append((task, outcome))
    return results


async def _dispatch_read_task(task: TaskSpec, state: dict, read_dispatch: dict, writer) -> object:
    """One read task's dispatch, wrapped so its own operation_progress event
    fires the moment IT finishes — not only once every read task in the
    batch has finished (a single asyncio.gather over the raw dispatch calls
    would only resolve as a whole). Exception handling mirrors
    _commit_cluster: caught and returned in place of the outcome rather than
    propagated, so one read's failure can't affect another's."""
    try:
        outcome = await read_dispatch[task.type](state, task)
    except Exception as exc:
        logger.warning(
            "execute: task %s failed (%s: %s) — other tasks in this batch still complete",
            task.type, type(exc).__name__, exc,
        )
        outcome = exc
    writer(_progress_event(task, state, ok=not isinstance(outcome, Exception)))
    return outcome


async def execute(tasks: list[TaskSpec], state: dict) -> dict:
    from app.graph import query_catalog_node, retrieve_context_node

    read_dispatch = {TaskType.DATABASE_QUERY: query_catalog_node, TaskType.RETRIEVE_CONTEXT: retrieve_context_node}
    write_tasks = [t for t in tasks if t.type in _CONTEXT_TASK_TYPES]
    read_tasks = [t for t in tasks if t.type in read_dispatch]
    writer = _stream_writer()

    # Reads never wait on writes, even a connected one — they run
    # immediately alongside the resolve+cluster+commit pipeline below, not
    # after it (see the classifier-redesign plan's read-after-write
    # decision).
    reads_future = asyncio.gather(*(_dispatch_read_task(t, state, read_dispatch, writer) for t in read_tasks))

    if write_tasks:
        resolved_tasks = await asyncio.gather(*(_resolve_write_task(t, state) for t in write_tasks))
        clusters = _cluster(resolved_tasks)
        cluster_results = await asyncio.gather(*(_commit_cluster(cluster, state, writer) for cluster in clusters))
        write_results: list[tuple[TaskSpec, object]] = [item for cluster in cluster_results for item in cluster]
    else:
        write_results = []

    raw_read_results = await reads_future

    # write_results may already carry an Exception per task (caught inside
    # _commit_cluster, one per-cluster failure isolated from the rest of
    # that cluster); raw_read_results may too (caught inside
    # _dispatch_read_task) — both funnel through the same trace handling
    # here rather than each having its own copy of it. The warning log for
    # each failure already happened at the point of capture (_commit_cluster/
    # _dispatch_read_task), not repeated here.
    all_results: list[tuple[TaskSpec, object]] = list(write_results) + list(zip(read_tasks, raw_read_results))

    results: list[tuple[TaskSpec, dict]] = []
    failure_trace: list[TraceEntry] = []
    for task, outcome in all_results:
        if isinstance(outcome, Exception):
            failure_trace.append(
                TraceEntry(
                    node_name=f"execute:{task.type.value}",
                    output_summary=f"failed: {type(outcome).__name__}: {outcome}",
                    input_summary=(task.target or state.get("message", ""))[:200],
                    duration_ms=0,
                )
            )
            continue
        results.append((task, outcome))

    merged: dict = {"trace": list(failure_trace)}
    retrieved_parts: list[str] = []
    update_summaries: list[str] = []
    for task, result in results:
        merged["trace"].extend(result.get("trace", []))
        if task.type in _CONTEXT_TASK_TYPES:
            if result.get("update_summary"):
                update_summaries.append(result["update_summary"])
          
            if not merged.get("pending_question") and result.get("pending_question"):
                merged["pending_question"] = result["pending_question"]
                merged["question_generated"] = True
            if "pending_gap" not in merged and "pending_gap" in result:
                merged["pending_gap"] = result["pending_gap"]
            # Same first-result-wins convention as pending_gap above. Only
            # build_context_node ever returns this (delete_context_node never
            # creates a room, so its pre-write active_room_id — already set
            # into `state` by classify_intent_node before execute() ran — is
            # always correct as-is); this only matters as a fallback
            # correction for a task that just created a BRAND-NEW room (see
            # build_context_node's own docstring), which is rare enough that
            # "first task in results order that has one" is an acceptable
            # tie-break rather than something worth reasoning about
            # cross-cluster commit ordering for.
            if not merged.get("active_room_id") and result.get("active_room_id"):
                merged["active_room_id"] = result["active_room_id"]
        elif result.get("retrieved"):
            retrieved_parts.append(result["retrieved"])

    if update_summaries:
        merged["update_summary"] = " ".join(update_summaries)
    if retrieved_parts:
        merged["retrieved"] = "\n\n".join(retrieved_parts)
    if any(t.type in _CONTEXT_TASK_TYPES for t in tasks):
        merged["needs_answer"] = True
    return merged


async def merge_task_results(parts: list[str]) -> str:
    """Composes multiple deterministic/generated pieces from one multi-task
    turn (e.g. a fact-confirmation summary alongside a real answer) into ONE
    natural-language reply, instead of surfacing them as separate bot
    messages (today's actual behavior in app/chat.py — context_updated,
    wrapup, and ask_question are each their own event/message). Reuses
    llm.merge_response, the same fast/low-stakes model
    generate_wrapup_message already uses — not the full answer model.

    Not called from chat.py yet — see ARCHITECTURE_BASELINE.md's freeze
    rule; building the capability doesn't mean cutting the live turn's
    message shape over to it."""
    non_empty = [p for p in parts if p]
    if len(non_empty) <= 1:
        return non_empty[0] if non_empty else ""
    return await llm.merge_response(non_empty)
