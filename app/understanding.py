"""Pure intent-understanding layer — see ARCHITECTURE_BASELINE.md Phase 1.

understand() reads a message plus a minimal, already-serialized slice of
context (a pending-field key, conversation history text) and returns a
Meaning. It never touches Mongo, never receives a session/state object, and
has no side effects — testable with a plain string in, Meaning out.

understand() is a thin wrapper around app.llm.classify_operations — a
single structured-output call that both splits the message into independent
operations and labels each with one of 5 intents. This replaced an earlier
pipeline of classify_intent (single/rarely-double label) + three deterministic
guards + a regex room-splitter (segment_clauses) — see the
classifier-redesign-decisions memory for that migration's history; none of
that old code is left here."""

from pydantic import BaseModel

from app import llm
from app.context_builder import LEAF_TO_FIELD_NAME
from app.models import FIELD_LABELS
from app.tasks import TaskSpec, TaskType

# Mechanical relabeling of classify_operations' 5 intent strings onto the
# existing TaskType vocabulary — a full bijection over every TaskType.
_INTENT_TO_TASK_TYPE: dict[str, TaskType] = {
    "CONTEXT_UPDATE": TaskType.EDIT_CONTEXT,
    "CONTEXT_DELETE": TaskType.DELETE_CONTEXT,
    "CONTEXT_RETRIEVAL": TaskType.RETRIEVE_CONTEXT,
    "DATABASE_RETRIEVAL": TaskType.DATABASE_QUERY,
    "DIRECT_ANSWER": TaskType.ANSWER,
}
TASK_TYPE_TO_INTENT: dict[TaskType, str] = {v: k for k, v in _INTENT_TO_TASK_TYPE.items()}


class Meaning(BaseModel):
    tasks: list[TaskSpec]
    raw_message: str
    # classify_operations' per-operation intents, in the same order as
    # `tasks` — telemetry only (app.graph.classify_intent_node's trace line).
    raw_intents: list[str]


def _pending_field_label(field_keys: list[str] | None) -> str | None:
    """Human-readable version of the currently-pending canonical path(s)
    (e.g. 'Project.Rooms.r1.Style' -> 'style') for classify_operations'
    prompt — joined into one comma-separated string when a batched question
    (app.question_engine.KnowledgeGapBatch) covered more than one field at
    once. A canonical_path's last segment is the PascalCase node_type (e.g.
    'Style', 'SquareFootage'), not a FIELD_LABELS key directly — routed
    through LEAF_TO_FIELD_NAME first, same node_type -> field_name table
    app.question_engine/app.graph already use for the same lookup."""
    if not field_keys:
        return None
    labels = []
    for key in field_keys:
        node_type = key.split(".")[-1]
        field_name = LEAF_TO_FIELD_NAME.get(node_type, node_type)
        labels.append(FIELD_LABELS.get(field_name, field_name))
    return ", ".join(dict.fromkeys(labels))


def _operations_to_tasks(operations: list["llm.Operation"]) -> list[TaskSpec]:
    """Operation -> TaskSpec, shared by understand()'s normal classify path
    and app.graph's pending_operation_questions resume path (which rebuilds
    tasks from a stashed operation list plus merged answers, without calling
    classify_operations again — see that branch's docstring).

    No longer derives a room_hint here (the static-vocabulary regex detector
    that used to — see the classifier-redesign-decisions memory — is
    removed as of the classifier-connection plan's Phase 5 cleanup):
    op.connection, sourced from classify_operations' own project-grounded
    CURRENT DATA TREE reasoning, strictly subsumes what regex vocabulary
    matching against static room names could ever offer. TaskSpec.room_hint
    the field still exists and is still consumed downstream (app.execution.
    _resolve_write_task, app.graph's delete/build_context paths) — just fed
    exclusively from connection now, via canonical_mapper.split_connection,
    never auto-detected here."""
    return [
        TaskSpec(
            type=_INTENT_TO_TASK_TYPE[op.intent],
            op_id=op.id,
            target=op.text,
            connection=op.connection,
        )
        for op in operations
    ]


async def understand(
    message: str,
    history: str = "",
    pending_field: list[str] | None = None,
    *,
    tree_text: str = "Project",
    capture: dict | None = None,
) -> Meaning:
    """`tree_text` (app.context_builder.render_project_tree_text's output)
    grounds classify_operations' `connection` output against the project's
    live state. Threaded through as a plain string, not fetched here —
    understand() stays DB-free/pure (see module docstring); app.graph.
    classify_intent_node is what actually queries the graph."""
    operations = await llm.classify_operations(
        message, history, pending_field=_pending_field_label(pending_field), tree_text=tree_text, capture=capture
    )
    # RULE 15 (classify_operations' own prompt): unclear/no-op input returns
    # an empty operations list. Falling back to a single DIRECT_ANSWER over
    # the full message guarantees the turn still gets a reply (e.g. a plain
    # "hello") instead of silently doing nothing.
    if not operations:
        operations = [llm.Operation(id="op_1", text=message, intent="DIRECT_ANSWER")]

    tasks = _operations_to_tasks(operations)
    raw_intents = [op.intent for op in operations]
    return Meaning(tasks=tasks, raw_message=message, raw_intents=raw_intents)
