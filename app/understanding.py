"""Pure intent-understanding layer — see ARCHITECTURE_BASELINE.md Phase 1.

understand() reads a message plus a minimal, already-serialized slice of
context (a pending-field key, conversation history text) and returns a
Meaning. It never touches Mongo, never receives a session/state object, and
has no side effects — testable with a plain string in, Meaning out.

guard_database_query/guard_split_intents moved here from app/graph.py
verbatim (still re-exported from there for backward compatibility — see
tests/test_graph.py) since they're part of turning the classifier's raw
output into a real Meaning, not part of graph wiring."""

import re

from pydantic import BaseModel

from app import deepinfra
from app.models import FIELD_LABELS
from app.tasks import TaskSpec, TaskType

# Mechanical relabeling of today's four intent strings — see app/tasks.py's
# docstring for why SAVE_CONTEXT/DELETE_CONTEXT aren't part of this map yet.
_INTENT_TO_TASK_TYPE: dict[str, TaskType] = {
    "update_context": TaskType.EDIT_CONTEXT,
    "direct_question": TaskType.ANSWER,
    "database_query": TaskType.DATABASE_QUERY,
    "context_related": TaskType.RETRIEVE_CONTEXT,
}
TASK_TYPE_TO_INTENT: dict[TaskType, str] = {v: k for k, v in _INTENT_TO_TASK_TYPE.items()}


class Meaning(BaseModel):
    tasks: list[TaskSpec]
    raw_message: str
    # classify_intent's output before either guard ran — kept so callers can
    # still report the "guard adjusted from: ..." trace telemetry that
    # app.graph.classify_intent_node relied on before this module existed.
    raw_intents: list[str]


def _pending_field_label(field_key: str | None) -> str | None:
    """Human-readable version of a raw pending_field key (e.g. 'room:<id>.style'
    -> 'style') for classify_intent's prompt."""
    if not field_key:
        return None
    field_name = field_key.split(".")[-1]
    return FIELD_LABELS.get(field_name, field_name)


# Second, deterministic signal in front of query_catalog's vector search —
# classify_intent alone over-fires database_query on messages that mention
# products only in passing. Requiring a trigger word too costs nothing (a
# regex, not a model call) and fails in the safe direction: a genuine
# retrieval request without a trigger word just gets answered directly
# instead of triggering an unnecessary catalog lookup. Two separate
# vocabularies (pricing vs. plain catalog search) since database_query's own
# definition in the prompt covers both "find/recommend a product" and
# "what does this cost" — a pricing-only trigger list silently killed the
# former, which is the more common phrasing.
_PRICING_TRIGGERS = re.compile(
    r"\b(price|pricing|cost|quote|quotation|how much|sku|invoice|catalog)\b", re.IGNORECASE
)
_SEARCH_TRIGGERS = re.compile(
    r"\b(recommend|suggest|find|show me|looking for|options for|"
    r"something like|style of|similar to)\b",
    re.IGNORECASE,
)


def guard_database_query(intents: list[str], message: str) -> list[str]:
    if "database_query" in intents and not (
        _PRICING_TRIGGERS.search(message) or _SEARCH_TRIGGERS.search(message)
    ):
        intents = [i for i in intents if i != "database_query"] or ["direct_question"]
    return intents


# Third guard, applied after guard_database_query so a stripped
# database_query never reaches this check — validates that a multi-label
# result from classify_intent is a genuine compound turn (a real question
# plus a separate connector-joined clause) rather than the model over-
# splitting a single fact-stating message. Matters beyond labeling
# correctness: handle_split_intents_node sets needs_answer=True whenever a
# second label rides along with update_context, which forces a full extra
# generate_answer LLM call — an over-split silently doubles generation cost
# on what should have been a plain, answer-free update_context turn.
_QUESTION_MARKERS = re.compile(r"\?|\b(what|how|when|why|should|can you|which|do you)\b", re.IGNORECASE)
_JOIN_MARKERS = re.compile(r"\band\b|,|\balso\b|\bplus\b|\bbut\b", re.IGNORECASE)


def guard_split_intents(intents: list[str], message: str) -> list[str]:
    if len(intents) <= 1:
        return intents

    has_question = bool(_QUESTION_MARKERS.search(message))
    has_join = bool(_JOIN_MARKERS.search(message))

    if has_question and has_join:
        return intents
    if has_question:
        intents = [i for i in intents if i != "update_context"]
    else:
        intents = ["update_context"] if "update_context" in intents else intents[:1]

    return intents or ["direct_question"]


# Deterministic override applied AFTER TaskType mapping, not a 5th classifier
# label — "delete_context" as its own intent string would need retraining/
# re-prompting classify_intent for a distinction a keyword list already
# catches reliably, same posture as guard_database_query's trigger-word gate
# above. classify_intent already treats a removal statement as update_context
# (it states a fact about the project — that something should no longer be
# there); this guard is what tells that apart from a value edit before
# extraction ever runs, which is exactly what previously produced a
# hallucinated field edit for a removal request.
_DELETE_TRIGGERS = re.compile(
    r"\b(remove|delete|cancel|get rid of|take out|drop the|don'?t need|scratch the|no longer)\b",
    re.IGNORECASE,
)


def guard_delete(task_type: TaskType, text: str) -> TaskType:
    if task_type == TaskType.EDIT_CONTEXT and _DELETE_TRIGGERS.search(text):
        return TaskType.DELETE_CONTEXT
    return task_type


class Clause(BaseModel):
    text: str
    # A raw room-TYPE NAME (e.g. "living room"), not a resolved room_id — see
    # app.tasks.TaskSpec.room_hint's own docstring for why this module can
    # never produce anything more resolved than a name (no DB access here).
    room_hint: str | None = None


# Static, not project-derived: a per-project "rooms that already exist"
# vocabulary would miss the common case of a message naming TWO BRAND NEW
# rooms on the very first turn, before either has a KnowledgeNode yet — which
# is exactly when a multi-room message is most likely to need segmenting.
# Longest-first alternation so "master bedroom"/"living room" match before
# the bare "bedroom"/"room" would (rapidfuzz isn't used here — this is exact
# vocabulary matching, not fuzzy similarity; app.graph._resolve_room_hint is
# where the fuzzy match against a specific project's real rooms happens).
_ROOM_VOCABULARY = sorted(
    [
        "master bedroom", "living room", "dining room", "family room", "guest room", "guest bedroom",
        "walk-in closet", "powder room", "laundry room", "mud room", "sun room", "kitchen", "bedroom",
        "bathroom", "office", "study", "nursery", "den", "basement", "attic", "garage", "hallway",
        "foyer", "pantry", "closet", "balcony", "patio",
    ],
    key=len,
    reverse=True,
)
_ROOM_MENTION_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in _ROOM_VOCABULARY) + r")\b", re.IGNORECASE)
_LEADING_CONNECTOR_RE = re.compile(r"^\s*(?:" + _JOIN_MARKERS.pattern + r")\s*", re.IGNORECASE)


def segment_clauses(message: str) -> list[Clause]:
    """Splits `message` into one Clause per distinct room it mentions, so a
    turn describing several rooms at once ("...to the living room and let's
    update the kitchen with...") gets written to each room's own subtree
    instead of all landing under whichever room was last active — see the
    deletion-support plan's Fix 2. Returns a single Clause(message,
    room_hint=None) — today's unsegmented behavior, unchanged — whenever
    fewer than 2 distinct rooms are named.

    Splits right before each NEW room's first mention, not on every join
    word: "sofa and couch" between two mentions of the SAME room never
    triggers a split, only switching to a genuinely different room does. A
    repeated later mention of an already-seen room doesn't open a new
    boundary either — it's folded into whichever clause is current at that
    point, a deliberate simplification of the much rarer "back to a room
    already covered" case. Clause text is NOT pretty-printed (a leading
    connector word is stripped, nothing else) — extract_fields tolerates
    imperfect phrasing fine; getting the ROOM ATTRIBUTION right is what
    actually matters here."""
    seen: set[str] = set()
    boundaries: list[tuple[str, int]] = []
    for m in _ROOM_MENTION_RE.finditer(message):
        name = m.group(1).lower()
        if name not in seen:
            seen.add(name)
            boundaries.append((name, m.start()))

    if len(boundaries) < 2:
        return [Clause(text=message)]

    split_points = [start for _, start in boundaries[1:]]
    pieces: list[str] = []
    prev = 0
    for point in split_points:
        pieces.append(message[prev:point])
        prev = point
    pieces.append(message[prev:])

    clauses: list[Clause] = []
    for i, (room_name, _) in enumerate(boundaries):
        piece = pieces[i]
        text = _LEADING_CONNECTOR_RE.sub("", piece, count=1).strip() if i > 0 else piece.strip()
        clauses.append(Clause(text=text or piece.strip(), room_hint=room_name))
    return clauses


async def _classify_clause(
    text: str, history: str, pending_field: str | None, room_hint: str | None, *, capture: dict | None
) -> tuple[list[TaskSpec], list[str]]:
    """One clause's worth of classify_intent + both existing guards +
    guard_delete — the exact single-clause pipeline understand() has always
    run, just parameterized so the multi-clause path below can run it once
    per clause instead of duplicating it. Returns (tasks, raw) — raw is
    classify_intent's own output BEFORE either guard ran, matching
    Meaning.raw_intents' existing contract (see its docstring — the
    "guard adjusted from: ..." trace telemetry classify_intent_node reports
    needs the pre-guard value, not what guard_split_intents/guard_delete
    settled on)."""
    raw = await deepinfra.classify_intent(text, history, pending_field=_pending_field_label(pending_field), capture=capture)
    # Order matters: guard_database_query first, so a database_query that it
    # strips never reaches guard_split_intents' two-label check.
    intents = guard_database_query(raw, text)
    intents = guard_split_intents(intents, text)
    tasks = [
        TaskSpec(type=guard_delete(_INTENT_TO_TASK_TYPE[i], text), target=text if room_hint else None, room_hint=room_hint)
        for i in intents
        if i in _INTENT_TO_TASK_TYPE
    ]
    return tasks, raw


async def understand(
    message: str,
    history: str = "",
    pending_field: str | None = None,
    *,
    capture: dict | None = None,
) -> Meaning:
    clauses = segment_clauses(message)

    if len(clauses) == 1:
        # Single-clause path — byte-for-byte the one classify_intent call
        # this function has always made; room_hint=None here means
        # target/room_hint both stay None on every resulting TaskSpec (see
        # _classify_clause), so every downstream node falls back to
        # state["message"]/state["active_room_id"] exactly as before
        # segmentation existed.
        tasks, raw_intents = await _classify_clause(message, history, pending_field, None, capture=capture)
        if not tasks:
            tasks = [TaskSpec(type=TaskType.ANSWER)]
        return Meaning(tasks=tasks, raw_message=message, raw_intents=raw_intents)

    # Multi-room turn: one classify_intent call per clause, each producing
    # its own TaskSpec(s) carrying that clause's own text/room (see
    # app.tasks.TaskSpec.target/room_hint) — reuses the exact same
    # classify+guard pipeline as the single-clause path, just applied per
    # clause instead of once for the whole message. `capture` (LLM input/
    # output for the trace) reflects only the LAST clause's call if more
    # than one needs capturing — an accepted trade-off, since the trace's
    # job is per-node debugging, not reconstructing every underlying call
    # of a multi-clause turn.
    all_tasks: list[TaskSpec] = []
    all_raw_intents: list[str] = []
    for clause in clauses:
        tasks, raw_intents = await _classify_clause(clause.text, history, pending_field, clause.room_hint, capture=capture)
        all_tasks.extend(tasks)
        all_raw_intents.extend(raw_intents)
    if not all_tasks:
        all_tasks = [TaskSpec(type=TaskType.ANSWER)]
    return Meaning(tasks=all_tasks, raw_message=message, raw_intents=all_raw_intents)
