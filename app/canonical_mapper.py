"""Canonical Mapping Engine — Phase 5/6. Resolves a freeform entity mention
(the kind today's extract_graph_links produces as a "preference"/
"constraint"/"attribute"/"entity" node) to one canonical KnowledgeNode:
reusing an existing instance when the mention is a synonym/alias of one
already captured ("TV Cabinet" -> the same node as an existing "TV Unit"),
and inferring the right ontology subtree (Materials vs. Furniture vs.
Attributes vs. Constraints vs. ClientPreferences) when it isn't. See
ontology/PHASE5_CANONICAL_MAPPER.md for the design writeup and
ontology/PHASE3_ONTOLOGY.md (Gaps 1, 2, 3, 5) for what this resolves.

Scope: only the five freeform, instantiable "fact bucket" types
(Materials/Furniture/Attributes/Constraints/ClientPreferences) go through
this mapper. Structured slot fields (ProjectType, Budget, Style, RoomType,
SquareFootage, ExistingFurniture, Timeline) are populated by
app.llm.extract_fields's already-deterministic schema — there's no
ambiguity there for a mapper to resolve, so routing them through
map_to_canonical() would be solving a problem that doesn't exist for them.

Never invents a canonical_path outside ontology/v1.yaml — every path this
module produces is built from real ontology/v1.yaml keys plus opaque
instance-id segments (room ids, slugified entity names), and a mention that
can't be confidently classified lands in the reviewed Unmapped bucket
instead of a made-up location.

Not wired into the live turn yet — that's Phase 7's job (see
ARCHITECTURE_BASELINE.md's freeze rule). Built and tested standalone, same
as app/execution.py was in Phase 1.

Embedding reuse (Phase 6): a mapping call embeds the query text once (never
at all if step 1 — exact string match — already resolves it) and reuses each
existing candidate's persisted KnowledgeNode.embedding instead of
re-embedding unchanging text on every call; the five ontology type
descriptions are static for the process lifetime, so they're embedded once
and cached module-wide rather than per call."""

import asyncio
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel

from app import llm, graph_store
from app.models import KnowledgeNode
from app.versioning import record_version

logger = logging.getLogger(__name__)

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent / "ontology" / "v1.yaml"


def _load_ontology() -> dict[str, Any]:
    with open(_ONTOLOGY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ONTOLOGY = _load_ontology()

# The only node types this mapper ever classifies a freeform mention into —
# see the module docstring's Scope note. "Unmapped" is the below-threshold
# fallback, not a type-inference target itself.
_FREEFORM_NODE_TYPES = ["Materials", "Furniture", "Attributes", "Constraints", "ClientPreferences"]
_UNMAPPED_TYPE = "Unmapped"
_LABEL_NODE_TYPE = "Label"

# Two distinct questions share this one threshold for now (see
# ontology/PHASE5_CANONICAL_MAPPER.md): (1) is an existing node the SAME
# entity as this mention (alias reuse), and (2) is the best-scoring ontology
# type confident enough to trust (vs. falling back to Unmapped). Not tuned
# against real embeddings — a documented starting point, not a measured one.
_MATCH_THRESHOLD = 0.75

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str) -> str:
    """Public (not `_`-prefixed): also used by app/context_builder.py
    (Phase 7) for Materials item path segments."""
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug or "item"


class CanonicalMatch(BaseModel):
    canonical_path: str
    node_type: str
    # Empty only for matched_via="no_match" (active_only=True search that found
    # nothing to point at) — every other case is a real resolved instance's
    # node_id, never the Label leaf's.
    node_id: str
    created_new: bool
    confidence: float
    matched_via: Literal["alias_exact", "alias_embedding", "type_hint", "type_embedding", "unmapped", "no_match"]
    flagged_for_review: bool


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Project's own top-level children (BasicInformation/Budget/Timeline/Rooms/
# Requirements/Quotation/Unmapped), read straight off the already-loaded
# ontology rather than a hand-duplicated list — a real path's first
# root-relative segment is always one of these (see is_grounded_connection).
_ONTOLOGY_TOP_LEVEL = set(_ONTOLOGY["Project"]["children"])
_CONNECTION_ROOM_RE = re.compile(r"^Rooms\.([^.]+)")


def is_grounded_connection(connection: str) -> bool:
    """True iff `connection` (app.llm.Operation.connection /
    app.tasks.TaskSpec.connection) is a real, root-relative path into the
    live tree rather than a free-text new-room/new-entity name — classify_
    operations' prompt renders paths root-relative (e.g. "Rooms.a1b2c3d4",
    never "Project.Rooms.a1b2c3d4", matching app.context_builder.
    render_project_tree_text's own convention), so a grounded connection's
    first segment is always one of Project's own ontology children. The
    literal "Project" sentinel (project-wide operations) is grounded too,
    even though "Project" is not itself one of its own children."""
    return connection == "Project" or connection.split(".", 1)[0] in _ONTOLOGY_TOP_LEVEL


def room_id_from_connection(connection: str) -> Optional[str]:
    """The room id out of a grounded connection like "Rooms.a1b2c3d4" or
    "Rooms.a1b2c3d4.Materials.countertop" — None for anything else
    (project-level "Project", a different top-level section, or an
    ungrounded free-text connection)."""
    match = _CONNECTION_ROOM_RE.match(connection)
    return match.group(1) if match else None


def split_connection(connection: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """(room_id_override, room_hint_override) derived from a classifier
    connection — shared by app.execution._resolve_write_task (EDIT_CONTEXT)
    and app.graph's delete-target resolution (DELETE_CONTEXT). A grounded
    connection's room segment, if any, becomes an exact room_id override; an
    ungrounded (free-text new-room/new-entity name) connection becomes a
    room_hint override — generalizing app.tasks.TaskSpec.room_hint's old
    regex-detected-vocabulary source to the classifier's own
    project-grounded guess. connection is always an anchor hint feeding the
    existing resolve pipeline, never a final write path — see
    is_grounded_connection's docstring and the classifier-connection plan."""
    if not connection:
        return None, None
    if is_grounded_connection(connection):
        return room_id_from_connection(connection), None
    return None, connection


def _container_path(node_type: str, room_id: Optional[str]) -> Optional[str]:
    """canonical_path of the container this node_type's instances live
    under. Returns None when node_type needs a room to be scoped under and
    none was given — the caller's cue to fall back to Unmapped (Gap 5)."""
    if node_type in ("Materials", "Furniture", "Attributes"):
        return f"Project.Rooms.{room_id}.{node_type}" if room_id else None
    if node_type in ("Constraints", "ClientPreferences"):
        return f"Project.Requirements.{node_type}"
    if node_type == _UNMAPPED_TYPE:
        return f"Project.Rooms.{room_id}.Unmapped" if room_id else "Project.Unmapped"
    raise ValueError(f"{node_type!r} is not a freeform instantiable type")


async def _existing_candidates(
    project_id: str, room_id: Optional[str], node_type_hint: Optional[str], *, active_only: bool = False
) -> list[tuple[KnowledgeNode, KnowledgeNode]]:
    """(label_node, instance_node) pairs for every existing freeform fact in
    this project, optionally narrowed to one room and/or one node type — the
    pool checked for an alias/embedding match before anything new is
    created. Fetches the whole project's nodes in one query and filters in
    Python rather than a $vectorSearch-style index (unlike app/rag.py's
    catalog search): a single project's freeform-fact count is small and
    bounded by nature, unlike an open-ended product catalog, so there's no
    real scale case for Atlas-only infrastructure here — see
    ontology/PHASE5_CANONICAL_MAPPER.md.

    active_only is opt-in, not the default: normal creation matching (the
    slug-collision check in _create_instance) still needs to see a retracted
    node's Label so a re-stated entity doesn't collide with one it should be
    considered distinct from. Only deletion targeting (app.graph.delete_context_node)
    passes active_only=True — matching a retraction request against an
    already-retracted node would let "remove the ceiling fan" silently
    no-op on a repeat, or worse, resurrect the wrong history.

    A real graph pattern match (Label -[:CHILD_OF]-> Instance, see
    app/graph_store.py) rather than the old "fetch every node in the
    project, build a by_id dict, filter in Python" — instance.node_type is
    already constrained to a freeform type by that pattern match (only a
    freeform instance ever parents a Label), so the old
    `instance.node_type not in _FREEFORM_NODE_TYPES` check isn't needed
    here either."""
    return await graph_store.find_label_instance_pairs(project_id, room_id, node_type_hint, active_only=active_only)


def _exact_alias_match(
    raw_entity: str, candidates: list[tuple[KnowledgeNode, KnowledgeNode]]
) -> Optional[tuple[KnowledgeNode, KnowledgeNode]]:
    needle = raw_entity.strip().lower()
    for label, instance in candidates:
        texts = [str(label.value or "")] + list(label.aliases)
        if any(t.strip().lower() == needle for t in texts):
            return label, instance
    return None


async def _score_candidates(
    query_vec: list[float], candidates: list[tuple[KnowledgeNode, KnowledgeNode]]
) -> list[tuple[tuple[KnowledgeNode, KnowledgeNode], float]]:
    """Every candidate scored against query_vec, best first — backfilling
    (embedding + saving) any candidate that doesn't have one yet (e.g. a
    Label created before this field existed) in one batched call rather than
    re-embedding every candidate on every call. Shared by _best_embedding_match
    (top score only, the normal match-or-create path) and
    resolve_deletion_target below (the full ranking, since deletion needs to
    tell "one confident match" apart from "two near-tied matches" before
    acting on either)."""
    if not candidates:
        return []

    missing = [(label, instance) for label, instance in candidates if not label.embedding]
    if missing:
        vectors = await llm.embed([str(label.value or "") for label, _ in missing])
        for (label, _instance), vector in zip(missing, vectors):
            label.embedding = vector
            await graph_store.save_node(label)

    scored = [(pair, _cosine_similarity(query_vec, pair[0].embedding)) for pair in candidates]
    return sorted(scored, key=lambda item: item[1], reverse=True)


async def _best_embedding_match(
    query_vec: list[float], candidates: list[tuple[KnowledgeNode, KnowledgeNode]]
) -> tuple[Optional[tuple[KnowledgeNode, KnowledgeNode]], float]:
    scored = await _score_candidates(query_vec, candidates)
    if not scored:
        return None, 0.0
    return scored[0]


_type_description_cache: Optional[dict[str, list[float]]] = None
_type_description_cache_lock = asyncio.Lock()


async def _get_type_description_vectors() -> dict[str, list[float]]:
    """The five freeform types' ontology/v1.yaml descriptions never change
    for the life of the process, so they're embedded once and cached rather
    than re-embedded on every map_to_canonical() call that needs
    type-inference."""
    global _type_description_cache
    if _type_description_cache is not None:
        return _type_description_cache
    async with _type_description_cache_lock:
        if _type_description_cache is None:
            descriptions = [_ONTOLOGY[t]["description"] for t in _FREEFORM_NODE_TYPES]
            vectors = await llm.embed(descriptions)
            _type_description_cache = dict(zip(_FREEFORM_NODE_TYPES, vectors))
    return _type_description_cache


async def _best_type_match(query_vec: list[float]) -> tuple[str, float]:
    type_vectors = await _get_type_description_vectors()
    scored = [(node_type, _cosine_similarity(query_vec, vec)) for node_type, vec in type_vectors.items()]
    best_type, best_score = max(scored, key=lambda pair: pair[1])
    return best_type, best_score


async def ensure_path(canonical_path: str, project_id: str) -> KnowledgeNode:
    """Get-or-create every node along canonical_path, in order, returning the
    deepest one. A segment matching an ontology/v1.yaml key is typed as
    itself (e.g. "Rooms", "Furniture"); any other segment is a room instance
    id directly under "Project.Rooms" (mirrors scripts/migrate_partial_context_to_tree.py's
    convention: the container "Project.Rooms" and each room instance
    "Project.Rooms.<room_id>" are both node_type="Rooms" — container vs.
    instance is a path-depth distinction, not a type distinction).

    Public (not `_`-prefixed): shared with app/context_builder.py (Phase 7),
    which needs the same get-or-create-ancestors behavior for structured
    field leaves, not just this module's freeform instances."""
    segments = canonical_path.split(".")
    parent: Optional[KnowledgeNode] = None
    current_parts: list[str] = []
    for segment in segments:
        current_parts.append(segment)
        path = ".".join(current_parts)
        existing = await graph_store.find_one(project_id, path)
        if existing is None:
            node_type = segment if segment in _ONTOLOGY else "Rooms"
            if parent is not None and parent.canonical_path == "Project.Rooms":
                room_id = segment  # this segment IS a room instance id
            else:
                room_id = parent.room_id if parent else None
            existing = KnowledgeNode(
                canonical_path=path, node_type=node_type, parent_id=parent.node_id if parent else None,
                project_id=project_id, room_id=room_id,
            )
            await graph_store.insert_node(existing)
        parent = existing
    return parent


async def _resolve_instance_path(
    raw_entity: str, node_type: str, project_id: str, room_id: Optional[str]
) -> tuple[str, KnowledgeNode]:
    """The collision-safe path a new freeform instance for raw_entity would
    live at, plus its (get-or-created) ancestor container — read-mostly:
    the only write is ensure_path's idempotent container scaffolding (e.g.
    Project.Rooms.<id>.Materials, no `value` of its own), never the instance
    or Label leaf itself. Shared by _create_instance (which then writes
    those leaves) and map_to_canonical's commit=False resolve path, so a
    previewed path and what actually gets created later can never disagree."""
    container_path = _container_path(node_type, room_id)
    container = await ensure_path(container_path, project_id)

    slug = slugify(raw_entity)
    instance_path = f"{container_path}.{slug}"
    if await graph_store.find_one(project_id, instance_path):
        # Two distinct mentions that happen to slugify the same (e.g. two
        # different "cabinet" entities) — disambiguate rather than collide.
        slug = f"{slug}_{uuid4().hex[:6]}"
        instance_path = f"{container_path}.{slug}"
    return instance_path, container


async def _create_instance(
    raw_entity: str,
    node_type: str,
    project_id: str,
    room_id: Optional[str],
    confidence: float,
    embedding: Optional[list[float]],
) -> KnowledgeNode:
    instance_path, container = await _resolve_instance_path(raw_entity, node_type, project_id, room_id)

    instance = KnowledgeNode(
        canonical_path=instance_path, node_type=node_type, parent_id=container.node_id,
        project_id=project_id, room_id=room_id, confidence=confidence,
    )
    await graph_store.insert_node(instance)

    label = KnowledgeNode(
        canonical_path=f"{instance_path}.Label", node_type=_LABEL_NODE_TYPE, parent_id=instance.node_id,
        value=raw_entity, project_id=project_id, room_id=room_id, confidence=confidence, embedding=embedding,
    )
    await graph_store.insert_node(label)
    # A newly-created Label always came from something the client said this
    # turn — see ontology/PHASE8_VERSIONING.md for why "inferred"/
    # "system_default" aren't used here (or anywhere yet).
    await record_version(label, changed_by="user_message")

    return instance


async def map_to_canonical(
    raw_entity: str,
    node_type_hint: Optional[str],
    project_id: str,
    room_id: Optional[str] = None,
    *,
    active_only: bool = False,
    commit: bool = True,
) -> CanonicalMatch:
    """room_id is optional but strongly recommended: without it, the three
    room-scoped freeform types (Materials/Furniture/Attributes) can never
    resolve to — or be created under — a specific room, and any such mention
    falls back to Unmapped (see ontology/PHASE3_ONTOLOGY.md, Gap 5).

    active_only=True is deletion's entry point (app.graph.delete_context_node):
    it restricts candidate search to live nodes (see _existing_candidates) and,
    critically, never falls through to the creation branches below — a
    retraction request that matches nothing gets matched_via="no_match", not a
    freshly-created Unmapped node standing in for the thing it was supposed to
    remove.

    commit=False (app.context_builder.resolve_context) previews what a
    creation branch WOULD do — the exact canonical_path it would create at
    (via _resolve_instance_path, the same collision-safe computation
    _create_instance itself uses) — without writing the instance/Label
    leaves, so a caller can learn a mention's target path for clustering
    purposes before any turn's writes are actually committed. node_id is ""
    on a commit=False creation preview, same convention as the no_match
    case, since nothing was created to have an id yet. An existing-node
    match (alias_exact/alias_embedding) is unaffected by commit — reusing an
    already-resolved node is not a "write pending commit," the alias-list
    bookkeeping on a re-stated synonym is idempotent and does not
    participate in delete/update/create ordering."""
    if node_type_hint is not None and node_type_hint not in _FREEFORM_NODE_TYPES:
        raise ValueError(f"node_type_hint must be one of {_FREEFORM_NODE_TYPES}, got {node_type_hint!r}")

    candidates = await _existing_candidates(project_id, room_id, node_type_hint, active_only=active_only)

    exact = _exact_alias_match(raw_entity, candidates)
    if exact is not None:
        _label, instance = exact
        return CanonicalMatch(
            canonical_path=instance.canonical_path, node_type=instance.node_type, node_id=instance.node_id,
            created_new=False, confidence=1.0, matched_via="alias_exact", flagged_for_review=False,
        )

    # Nothing beyond this point is answerable from stored data alone — every
    # remaining path needs raw_entity's own embedding, computed exactly once
    # and threaded through (never a second call for the same text). Skip it
    # entirely when active_only has nothing to compare against — there's
    # nothing to create on this path, so there's nothing an embedding could
    # still resolve.
    query_vec: Optional[list[float]] = None
    if candidates or (node_type_hint is None and not active_only):
        query_vec = (await llm.embed([raw_entity]))[0]

    if candidates:
        best_pair, best_score = await _best_embedding_match(query_vec, candidates)
        if best_pair is not None and best_score >= _MATCH_THRESHOLD:
            label, instance = best_pair
            normalized = raw_entity.strip()
            known = {str(label.value or "").strip().lower()} | {a.lower() for a in label.aliases}
            if normalized.lower() not in known:
                label.aliases.append(normalized)
                await graph_store.save_node(label)
            return CanonicalMatch(
                canonical_path=instance.canonical_path, node_type=instance.node_type, node_id=instance.node_id,
                created_new=False, confidence=best_score, matched_via="alias_embedding", flagged_for_review=False,
            )

    if active_only:
        return CanonicalMatch(
            canonical_path="", node_type=node_type_hint or "", node_id="",
            created_new=False, confidence=0.0, matched_via="no_match", flagged_for_review=False,
        )

    if node_type_hint is not None:
        node_type, type_confidence, matched_via = node_type_hint, 1.0, "type_hint"
    else:
        node_type, type_confidence = await _best_type_match(query_vec)
        matched_via = "type_embedding"

    if type_confidence < _MATCH_THRESHOLD or _container_path(node_type, room_id) is None:
        logger.warning(
            "map_to_canonical: %r in project %s could not be confidently placed "
            "(best type=%s score=%.2f, room_id=%s) — routed to Unmapped for review",
            raw_entity, project_id, node_type, type_confidence, room_id,
        )
        final_type, final_matched_via, final_flagged = _UNMAPPED_TYPE, "unmapped", True
    else:
        final_type, final_matched_via, final_flagged = node_type, matched_via, False

    if not commit:
        instance_path, _container = await _resolve_instance_path(raw_entity, final_type, project_id, room_id)
        return CanonicalMatch(
            canonical_path=instance_path, node_type=final_type, node_id="",
            created_new=True, confidence=type_confidence, matched_via=final_matched_via, flagged_for_review=final_flagged,
        )

    instance = await _create_instance(raw_entity, final_type, project_id, room_id, type_confidence, query_vec)
    return CanonicalMatch(
        canonical_path=instance.canonical_path, node_type=final_type, node_id=instance.node_id,
        created_new=True, confidence=type_confidence, matched_via=final_matched_via, flagged_for_review=final_flagged,
    )


# Stricter than _MATCH_THRESHOLD (0.75, used for creation/dedup matching) —
# creating a low-confidence node in Unmapped for later review is safe;
# retracting the wrong node on a low-confidence guess is not. Asymmetric
# risk, asymmetric threshold.
_DELETION_MATCH_THRESHOLD = 0.85
# Two candidates within this margin of each other (cosine similarity, 0-1
# scale) are treated as tied — "which ceiling fan?" territory, not "pick the
# best one and move on."
_DELETION_AMBIGUITY_MARGIN = 0.05


class DeletionCandidate(BaseModel):
    instance: KnowledgeNode
    # The freeform instance's own Label child's value (e.g. "ceiling fan") —
    # the instance node itself never carries a `value` (see _create_instance),
    # so this is what a caller should actually show the user, not
    # instance.value.
    label: str


class DeletionMatch(BaseModel):
    outcome: Literal["single", "ambiguous", "none"]
    match: Optional[DeletionCandidate] = None   # set only when outcome == "single"
    candidates: list[DeletionCandidate] = []     # set only when outcome == "ambiguous"
    confidence: float = 0.0


async def resolve_deletion_target(
    raw_entity: str, project_id: str, room_id: Optional[str] = None
) -> DeletionMatch:
    """The one entry point app.graph.delete_context_node uses to find what a
    retraction request refers to among LIVE (lifecycle="active") freeform
    facts. Deliberately separate from map_to_canonical: that function's job
    is match-or-CREATE and its single-best-match return shape can't express
    "two equally good candidates" — exactly the case deletion must not guess
    through (see the "ambiguous room hint" edge case in the deletion-support
    plan). Read-only: never records a new alias, never creates anything, even
    as a side effect — embedding backfill for an EXISTING candidate (via
    _score_candidates) is the only write this function can cause, same as
    map_to_canonical's own normal matching already does."""
    candidates = await _existing_candidates(project_id, room_id, None, active_only=True)
    if not candidates:
        return DeletionMatch(outcome="none")

    exact = _exact_alias_match(raw_entity, candidates)
    if exact is not None:
        label, instance = exact
        return DeletionMatch(
            outcome="single", match=DeletionCandidate(instance=instance, label=str(label.value or "")), confidence=1.0
        )

    query_vec = (await llm.embed([raw_entity]))[0]
    scored = await _score_candidates(query_vec, candidates)
    top_score = scored[0][1]
    if top_score < _DELETION_MATCH_THRESHOLD:
        return DeletionMatch(outcome="none", confidence=top_score)

    tied = [pair for pair, score in scored if score >= top_score - _DELETION_AMBIGUITY_MARGIN]
    if len(tied) > 1:
        return DeletionMatch(
            outcome="ambiguous",
            candidates=[DeletionCandidate(instance=instance, label=str(label.value or "")) for label, instance in tied],
            confidence=top_score,
        )

    label, instance = tied[0]
    return DeletionMatch(
        outcome="single", match=DeletionCandidate(instance=instance, label=str(label.value or "")), confidence=top_score
    )
