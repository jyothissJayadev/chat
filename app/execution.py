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

from app import deepinfra
from app.models import TraceEntry
from app.tasks import TaskSpec, TaskType

logger = logging.getLogger(__name__)


_CONTEXT_TASK_TYPES = (TaskType.EDIT_CONTEXT, TaskType.DELETE_CONTEXT)


async def execute(tasks: list[TaskSpec], state: dict) -> dict:
    from app.graph import build_context_node, delete_context_node, query_catalog_node, retrieve_context_node

    dispatch = {
        TaskType.EDIT_CONTEXT: build_context_node,
        TaskType.DELETE_CONTEXT: delete_context_node,
        TaskType.DATABASE_QUERY: query_catalog_node,
        TaskType.RETRIEVE_CONTEXT: retrieve_context_node,
    }

    # Dispatched per TaskSpec INSTANCE, not per distinct TaskType — a prior
    # version of this loop keyed both dispatch and the results merge by
    # TaskType, which silently dropped every task after the first of a
    # repeated type. Latent until app.understanding.segment_clauses started
    # producing two EDIT_CONTEXT/DELETE_CONTEXT tasks from one multi-room
    # message (Fix 2 of the deletion-support plan; see that plan's
    # Correction 3). Each dispatched node gets the same state dict plus its
    # own TaskSpec — build_context_node/delete_context_node read
    # task.target/task.room_hint in preference to
    # state["message"]/state["active_room_id"] when a task is passed (see
    # their own docstrings); the single-task direct-routing call sites in
    # app.graph.build_graph() still call these with no task argument, so
    # that path is unaffected.
    runnable = [t for t in tasks if t.type in dispatch]
    # return_exceptions=True (Phase 11 hardening): each dispatched branch
    # already has its own internal resilience (extract_fields/extract_graph_links
    # etc. skip their own failures rather than raising — see
    # app/context_builder.py), but a genuinely unexpected exception here must
    # still not take the OTHER branches in this batch down with it via a bare
    # asyncio.gather.
    raw_results = await asyncio.gather(*(dispatch[t.type](state, t) for t in runnable), return_exceptions=True)

    results: list[tuple[TaskSpec, dict]] = []
    failure_trace: list[TraceEntry] = []
    for task, outcome in zip(runnable, raw_results):
        if isinstance(outcome, Exception):
            logger.warning(
                "execute: task %s failed (%s: %s) — other tasks in this batch still complete",
                task.type, type(outcome).__name__, outcome,
            )
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
    # "Active room" after a multi-room turn is whichever room the LAST
    # context-writing task in the batch touched — matches the single-task
    # case (only one candidate ever existed there) and gives a reasonable
    # default for the next turn's follow-up question, rather than inventing a
    # multi-value-active-room concept nothing downstream is ready to consume.
    last_active_room_id = None
    for task, result in results:
        merged["trace"].extend(result.get("trace", []))
        if task.type in _CONTEXT_TASK_TYPES:
            if result.get("update_summary"):
                update_summaries.append(result["update_summary"])
            if "active_room_id" in result:
                last_active_room_id = result["active_room_id"]
            # First conflict/clarification in the batch wins — mirrors
            # build_context_node's own result.pending_confirmations[0]
            # behavior for multiple conflicts within a single extraction
            # call; the rest wait for a later turn rather than stacking
            # multiple unresolved questions onto one turn.
            if not merged.get("pending_confirmation") and result.get("pending_confirmation"):
                merged["pending_confirmation"] = result["pending_confirmation"]
                merged["question_generated"] = True
            if not merged.get("pending_question") and result.get("pending_question"):
                merged["pending_question"] = result["pending_question"]
                merged["question_generated"] = True
        elif result.get("retrieved"):
            retrieved_parts.append(result["retrieved"])

    if update_summaries:
        merged["update_summary"] = " ".join(update_summaries)
    if last_active_room_id is not None:
        merged["active_room_id"] = last_active_room_id
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
    deepinfra.merge_response, the same fast/low-stakes model
    generate_wrapup_message already uses — not the full answer model.

    Not called from chat.py yet — see ARCHITECTURE_BASELINE.md's freeze
    rule; building the capability doesn't mean cutting the live turn's
    message shape over to it."""
    non_empty = [p for p in parts if p]
    if len(non_empty) <= 1:
        return non_empty[0] if non_empty else ""
    return await deepinfra.merge_response(non_empty)
