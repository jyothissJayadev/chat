"""Task vocabulary produced by app.understanding.understand() and consumed by
app.execution.execute() — see ARCHITECTURE_BASELINE.md, Phase 1/2. Each
TaskType maps mechanically onto one of app.llm.classify_operations'
5 intent labels (see app.understanding._INTENT_TO_TASK_TYPE); this is a
relabeling of that classifier's own vocabulary, not a new classification
scheme.

SAVE_CONTEXT is deliberately not included yet: the current system never
distinguishes "first value for this field" from "edit to an existing value"
before extraction runs (see EDIT_CONTEXT) — that's still scoped as new work
for a later phase, not part of this relabeling. DELETE_CONTEXT (below) is
now built — see app.graph.delete_context_node and app.versioning.retract_node/
retract_subtree — the KnowledgeNode.lifecycle field it depends on replaces
the old field_status-only-moves-forward limitation this docstring used to
describe."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskType(str, Enum):
    ANSWER = "ANSWER"
    EDIT_CONTEXT = "EDIT_CONTEXT"
    DELETE_CONTEXT = "DELETE_CONTEXT"
    RETRIEVE_CONTEXT = "RETRIEVE_CONTEXT"
    DATABASE_QUERY = "DATABASE_QUERY"


class TaskSpec(BaseModel):
    type: TaskType
    # Mirrors app.llm.Operation.id ("op_1", "op_2", ...) — the key a
    # later clarifying-question reply's operation_answers dict is matched
    # against (see app.graph's pending_operation_questions branch). Defaults
    # to "" only for hand-built TaskSpecs in tests that don't exercise that
    # path; understanding.understand() always sets it.
    op_id: str = ""
    # This operation's own text, exactly as classify_operations split it out
    # (app.understanding.understand always sets this now — never None; a
    # single-operation turn's target is just that operation's own text,
    # which classify_operations' own RULE 3 keeps close to the original
    # message wording).
    target: Optional[str] = None
    # A raw room-TYPE NAME (e.g. "living room", "kitchen") — NOT a resolved
    # room_id; resolution happens downstream via the same fuzzy-match-or-create
    # path app.context_builder._resolve_room runs for extracted.roomType (see
    # build_context_node's room_hint handling). Never set by
    # understanding.understand() itself (no auto-detection exists anymore —
    # see the classifier-redesign-decisions and classifier-connection-plan
    # memories for that history); app.execution._resolve_write_task and
    # app.graph's delete/build_context paths derive this locally, per task,
    # from connection instead (see app.canonical_mapper.split_connection) —
    # this field stays on TaskSpec only as what those call sites pass
    # through, not as something a caller is expected to set directly.
    room_hint: Optional[str] = None
    # Mirrors app.llm.Operation.connection — the classifier's own
    # grounding guess for this operation (a root-relative canonical path, a
    # not-yet-existing entity name, or None). None here means the classifier
    # couldn't place it: app.graph's pending_operation_questions branch asks
    # the user directly rather than letting app.execution's resolve pass
    # guess. Once set (by the classifier, or by a clarifying-question
    # answer), app.execution._resolve_write_task treats it as an anchor hint
    # into context_builder.resolve_context — never a final write path (see
    # app.canonical_mapper.is_grounded_connection/room_id_from_connection).
    connection: Optional[str] = None
    # Mirrors app.llm.Operation.confusion/confusion_note — a genuine
    # unresolved either/or CONTENT decision the user stated but didn't
    # commit to (see app.prompts.CLASSIFY_OPERATIONS_SYSTEM_TEMPLATE's
    # CONFUSION section). Independent of connection: a task can be
    # ungrounded, content-undecided, both, or neither. app.graph's
    # clarifying-question branch holds the turn for confusion=True the same
    # way it does for connection=None.
    confusion: bool = False
    confusion_note: Optional[str] = None
