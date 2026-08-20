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


def legal_fields(node_type: str) -> list[str]:
    """The leaf field names ontology/v1.yaml actually allows under
    `node_type` (e.g. "Materials" -> ["Label", "Material", "Specification"])
    — empty for a node_type with no `fields` entry (a pure container type
    like "Rooms" or "Requirements"). Public: used by
    app.llm._validate_context_changes to check that a freeform entity edit's
    claimed field is legal for the entity's own node_type, the same
    already-loaded _ONTOLOGY this module uses everywhere else for
    type-inference/container-path decisions."""
    return _ONTOLOGY.get(node_type, {}).get("fields", [])


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
# Room ids are always minted as uuid4().hex[:8] (see app.pipeline._run_first_action's
# new-room branch) — 8 lowercase hex characters, never anything else. Matching
# that exact shape (rather than the old "any non-dot run" [^.]+) is what makes
# room_id_from_connection reject a leaked prompt-example placeholder like
# "<living_room_id>" instead of extracting it verbatim and using it to fabricate
# a real "Rooms.<living_room_id>" node — see the postmortem on session
# 36af868e for how that happened live.
_CONNECTION_ROOM_RE = re.compile(r"^Rooms\.([0-9a-f]{8})(?:\.|$)")


def to_full_path(root_relative_path: str) -> str:
    """Prepends the "Project." root segment to a root-relative path — the
    classifier's `connection`, or a resolver's `existing_path`/deletion
    target, all root-relative per the prompt/tree convention documented on
    is_grounded_connection below — to get the full canonical_path
    graph_store/Neo4j actually stores things under. Callers that query or
    write via graph_store (app.pipeline._apply_entity_edit,
    app.pipeline._apply_deletions) need this; callers stuck comparing
    against a root-relative set (app.llm._validate_context_changes) should
    use to_root_relative on the OTHER side instead of calling this at all —
    see that function's postmortem note for why "compare full against
    root-relative" was the actual bug, not a missing conversion here."""
    return f"Project.{root_relative_path}"


def to_root_relative(full_path: str) -> str:
    """Inverse of to_full_path — strips the "Project." root segment off a
    real canonical_path (as read from graph_store) to get the root-relative
    form the classifier/resolver prompts and their `connection`/
    `existing_path`/deletion-target fields use. The bare "Project" root node
    itself (never a valid existing_path/deletion target) is returned
    unchanged, same as any other path with no "Project." prefix to strip."""
    prefix = "Project."
    return full_path[len(prefix):] if full_path.startswith(prefix) else full_path


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


def split_connection(connection: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(room_id_override, room_hint_override, parent_instance_path) derived
    from a classifier connection — shared by app.pipeline._run_first_action
    (EDIT_CONTEXT) and app.graph's delete-target resolution (DELETE_CONTEXT).
    A grounded connection's room segment, if any, becomes an exact room_id
    override; an ungrounded (free-text new-room/new-entity name) connection
    becomes a room_hint override — generalizing app.tasks.TaskSpec.room_hint's
    old regex-detected-vocabulary source to the classifier's own
    project-grounded guess.

    `parent_instance_path` is the root-relative canonical path of an existing
    freeform instance the operation attaches to, when `connection` points
    DEEPER than `Rooms.<id>` (e.g. "Rooms.c785b3b6.Furniture.beds" — the bed
    a bedcover is being added TO). None when the connection is room-level or
    shallower, since there is no parent instance to attach parts to in that
    case. Threaded into canonical_mapper.map_part_to_canonical /
    map_property_to_canonical by app.pipeline._apply_update so a composite
    entity mention (bedcover, sofa cover, pillows) lands as a child of the
    named parent instance rather than as a sibling under the room's
    Furniture/Materials container. connection is always an anchor hint
    feeding the existing resolve pipeline, never a final write path — see
    is_grounded_connection's docstring and the classifier-connection plan."""
    if not connection:
        return None, None, None
    if is_grounded_connection(connection):
        room_id = room_id_from_connection(connection)
        # A "Rooms.something" connection whose room segment doesn't match the
        # real uuid4().hex[:8] id shape is not actually grounded — most often
        # a leaked prompt-example placeholder (e.g. "Rooms.<living_room_id>")
        # that an upstream classifier echoed instead of substituting a real
        # id. Treating this as room_hint=connection would be worse than doing
        # nothing: app.pipeline._run_first_action mints a brand-new room from
        # any non-None room_hint, so it would create a room whose RoomType is
        # the literal garbage string. Falling all the way through to
        # (None, None, None) instead sends the mention to the project-level
        # Unmapped bucket (see canonical_mapper._container_path) for manual
        # review — visible and inert, not a fabricated room.
        if connection.startswith("Rooms.") and room_id is None:
            return None, None, None
        # Deeper than "Rooms.<id>" => the tail beyond the room id is an
        # existing instance path the operation targets (e.g. "Furniture.beds").
        # Reconstruct the root-relative parent instance path; only set when
        # there genuinely is a tail (a bare "Rooms.<id>" connection has none).
        tail = connection.split(".", 2)[2] if connection.count(".") >= 2 else ""
        # connection is root-relative (no "Project." prefix, per the prompt
        # convention); graph_store canonical_paths carry the prefix, so prepend
        # it here to give callers a path usable directly with graph_store.
        parent_instance_path = f"Project.{connection}" if tail else None
        return room_id, None, parent_instance_path
    return None, connection, None


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
    itself (e.g. "Rooms", "Furniture"); any other segment inherits its
    parent's node_type. That covers two distinct cases with one rule: a room
    instance id directly under "Project.Rooms" (mirrors
    scripts/migrate_partial_context_to_tree.py's convention: the container
    "Project.Rooms" and each room instance "Project.Rooms.<room_id>" are both
    node_type="Rooms" — container vs. instance is a path-depth distinction,
    not a type distinction) — AND a freeform instance slug under a
    Materials/Furniture/Parts/Properties/... container (e.g.
    "...Materials.flooring" must be node_type="Materials", matching
    app/context_builder.py's `_is_container` assumption that a container
    shares its instances' node_type). Inheriting from parent handles both:
    "Project.Rooms" itself is node_type="Rooms" (an ontology key), so a room
    id under it inherits "Rooms"; a Materials container is node_type=
    "Materials", so an instance slug under it inherits "Materials". Previously
    hardcoded to "Rooms" for every non-ontology segment regardless of parent,
    which was only ever correct for the room-id case — see the CONTEXT TABLE
    plan for how this silently dropped every structured-extraction materials/
    furniture write from app.context_builder.render_project_tree_text.

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
            if segment in _ONTOLOGY:
                node_type = segment
            elif parent is not None:
                node_type = parent.node_type
            else:
                node_type = "Rooms"  # unreachable in practice: the first segment ("Project") is always an ontology key
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
    raw_entity: str, container_path: str, project_id: str
) -> tuple[str, KnowledgeNode]:
    """The collision-safe path a new freeform instance for raw_entity would
    live at, plus its (get-or-created) ancestor container — read-mostly:
    the only write is ensure_path's idempotent container scaffolding (e.g.
    Project.Rooms.<id>.Materials, no `value` of its own), never the instance
    or Label leaf itself. Shared by _create_instance (which then writes
    those leaves) and map_to_canonical's commit=False resolve path, so a
    previewed path and what actually gets created later can never disagree.

    Takes the container_path explicitly (rather than deriving it from
    node_type+room_id via _container_path) so the same code path serves both
    room-scoped freeform instances (Materials/Furniture/...) AND nested
    composite children (Parts under a parent Furniture instance, Properties
    under any instance) whose container is the parent instance path + the
    child container segment, not a function of node_type alone."""
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
    *,
    container_path: Optional[str] = None,
) -> KnowledgeNode:
    """Creates a freeform instance + its Label leaf. `container_path`
    defaults to the room-scoped container derived from node_type+room_id
    (the original Materials/Furniture/... case); pass it explicitly to
    create a composite child (Parts/Properties) under a parent instance
    path instead — the only thing that differs is which container the
    instance nests under, everything else (instance node, Label leaf,
    version row) is identical."""
    if container_path is None:
        container_path = _container_path(node_type, room_id)
    instance_path, container = await _resolve_instance_path(raw_entity, container_path, project_id)

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
        instance_path, _container = await _resolve_instance_path(raw_entity, _container_path(final_type, room_id), project_id)
        return CanonicalMatch(
            canonical_path=instance_path, node_type=final_type, node_id="",
            created_new=True, confidence=type_confidence, matched_via=final_matched_via, flagged_for_review=final_flagged,
        )

    instance = await _create_instance(raw_entity, final_type, project_id, room_id, type_confidence, query_vec)
    return CanonicalMatch(
        canonical_path=instance.canonical_path, node_type=final_type, node_id=instance.node_id,
        created_new=True, confidence=type_confidence, matched_via=final_matched_via, flagged_for_review=final_flagged,
    )


# ---------------------------------------------------------------------------
# Composite-entity children: Parts and Properties nested under a freeform
# instance (sofa -> cover/pillows, sofa -> color=red). These are NOT routed
# through the top-level map_to_canonical classification — Parts/Properties
# are deliberately NOT in _FREEFORM_NODE_TYPES, since they only ever attach
# to an already-resolved parent instance (no room-level type inference, no
# project-wide alias pool). Reuses the same alias-exact / embedding-dedup
# machinery as map_to_canonical, just scoped to the parent's child container.
# See the composite-entity plan (Parts/Properties ontology addition).
# ---------------------------------------------------------------------------


async def _child_candidates(
    parent_instance_path: str, child_container_segment: str, project_id: str
) -> list[tuple[KnowledgeNode, KnowledgeNode]]:
    """(label_node, instance_node) pairs for every existing child instance
    of `parent_instance_path` under the named container segment
    ("Parts"/"Properties") — the candidate pool checked before a new part/
    property is created, scoped to that one parent so a cover on sofa A
    never reuses sofa B's cover. Reuses graph_store.find_label_instance_pairs
    by filtering its full-project result to this parent's subtree, rather
    than adding a new Cypher path-prefix variant — the per-parent child
    count is tiny and a project's freeform-fact count is bounded (same
    reasoning as _existing_candidates)."""
    container_path = f"{parent_instance_path}.{child_container_segment}"
    pairs = await graph_store.find_label_instance_pairs(project_id, None, child_container_segment)
    return [
        (label, instance)
        for label, instance in pairs
        if instance.canonical_path.startswith(f"{container_path}.")
    ]


async def _map_child_to_canonical(
    raw_entity: str,
    child_container_segment: str,
    parent_instance_path: str,
    project_id: str,
    room_id: Optional[str],
    *,
    commit: bool = True,
) -> CanonicalMatch:
    """Shared resolve+create body for both Parts (raw_entity = "bedcover")
    and Properties (raw_entity = the property name, e.g. "color"). The only
    thing that differs between the two is `child_container_segment`
    ("Parts" vs "Properties") — the instance node_type is that segment, the
    container is `{parent_instance_path}.{segment}`, and the same alias-
    exact / embedding dedup against existing siblings runs before anything
    new is created. `commit=False` previews the would-be path without
    writing, same contract as map_to_canonical."""
    container_path = f"{parent_instance_path}.{child_container_segment}"
    candidates = await _child_candidates(parent_instance_path, child_container_segment, project_id)

    exact = _exact_alias_match(raw_entity, candidates)
    if exact is not None:
        _label, instance = exact
        return CanonicalMatch(
            canonical_path=instance.canonical_path, node_type=instance.node_type, node_id=instance.node_id,
            created_new=False, confidence=1.0, matched_via="alias_exact", flagged_for_review=False,
        )

    query_vec: Optional[list[float]] = None
    if candidates:
        query_vec = (await llm.embed([raw_entity]))[0]
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

    if not commit:
        instance_path, _container = await _resolve_instance_path(raw_entity, container_path, project_id)
        return CanonicalMatch(
            canonical_path=instance_path, node_type=child_container_segment, node_id="",
            created_new=True, confidence=1.0, matched_via="type_hint", flagged_for_review=False,
        )

    instance = await _create_instance(
        raw_entity, child_container_segment, project_id, room_id, 1.0, query_vec,
        container_path=container_path,
    )
    return CanonicalMatch(
        canonical_path=instance.canonical_path, node_type=child_container_segment, node_id=instance.node_id,
        created_new=True, confidence=1.0, matched_via="type_hint", flagged_for_review=False,
    )


async def map_part_to_canonical(
    raw_entity: str,
    parent_instance_path: str,
    project_id: str,
    room_id: Optional[str] = None,
    *,
    commit: bool = True,
) -> CanonicalMatch:
    """Resolve a part mention ("bedcover", "velvet cover") to a canonical
    Parts instance nested under `parent_instance_path` (e.g.
    Project.Rooms.<id>.Furniture.beds.Parts.bedcover). Reuses an existing
    sibling part on alias-exact/embedding match; otherwise creates a new
    Parts instance + Label leaf. One level only — a part's own sub-parts are
    never created here (the resolver folds them into the part's
    Properties/Quantity per the composite-entity plan)."""
    return await _map_child_to_canonical(raw_entity, "Parts", parent_instance_path, project_id, room_id, commit=commit)


async def map_property_to_canonical(
    name: str,
    value: str,
    parent_instance_path: str,
    project_id: str,
    room_id: Optional[str] = None,
    *,
    commit: bool = True,
) -> CanonicalMatch:
    """Resolve a named property ("color"="red") to a canonical Properties
    instance nested under `parent_instance_path`, writing both its Label
    (=name) and Value (=value) leaves. Reuses an existing sibling property of
    the same name on alias-exact match (updating its Value); otherwise
    creates a new Properties instance. `value` is written by the caller via
    the standard ProposedWrite path — this function only resolves the
    Properties instance + its Label, returning the canonical_path the Value
    leaf should be written at (`{match.canonical_path}.Value`)."""
    return await _map_child_to_canonical(name, "Properties", parent_instance_path, project_id, room_id, commit=commit)
