"""Task vocabulary produced by app.understanding.understand() and consumed by
app.execution.execute() — see ARCHITECTURE_BASELINE.md, Phase 1/2. Each
TaskType maps mechanically onto one of today's four intent labels (see
app.understanding._INTENT_TO_TASK_TYPE); this is a relabeling of the
existing classify_intent vocabulary, not a new classification scheme.

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
    # Clause/message text this task acts on — None means "the whole turn's
    # message" (today's single-clause behavior, still the default; see
    # app.understanding.segment_clauses). Populated per-clause once a message
    # is segmented into more than one task.
    target: Optional[str] = None
    # A raw room-TYPE NAME (e.g. "living room", "kitchen") — NOT a resolved
    # room_id. app.understanding.segment_clauses (pure, DB-free, same as the
    # rest of app.understanding) can only ever detect a room by name, never
    # resolve it to an existing project's actual room_id; that resolution
    # happens downstream, the same fuzzy-match-or-create path
    # app.context_builder._resolve_room already runs for extracted.roomType
    # (see build_context_node's room_hint handling) — a plain string, not a
    # pre-resolved id, is what a caller with no DB access can produce, and
    # what every consumer downstream actually needs to run through anyway.
    room_hint: Optional[str] = None
