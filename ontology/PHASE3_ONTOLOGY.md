# Phase 3 — Canonical Ontology: design doc and field-to-path mapping

Companion to `ontology/v1.yaml`. Per the implementation plan, Phase 3 is a
**design phase — no code merge until this is signed off**. Nothing in
`app/` reads `ontology/v1.yaml` yet; Phase 4 (Knowledge Tree data model) is
the first phase that depends on this being correct.

## 1. Field-to-path mapping table

Every field currently in `PartialContext` (see `ARCHITECTURE_BASELINE.md`)
and every node type currently in `ContextGraph` must have exactly one home
below — either a `canonical_path` in `ontology/v1.yaml`, or an explicit
"out of ontology" designation with a stated reason. Nothing is silently
unmapped.

| Source | Field / type | Canonical path | Notes |
|---|---|---|---|
| `ProjectMeta.projectType` | scalar | `Project.BasicInformation.ProjectType` | |
| `ProjectMeta.overallBudget` | scalar | `Project.Budget.Total` | |
| `ProjectMeta.timeline` | scalar | `Project.Timeline.Value` | |
| `ProjectMeta.moreRoomsPending` | scalar | **out of ontology** — `DialogueState` (Phase 7) | Pure interview-completeness flag ("has the client told us about every room"), not a fact about the project itself — nothing downstream (quotation, graph reasoning) needs to know this was asked. Same category as `field_attempts`, which the master plan already assigns to `DialogueState`. |
| `RoomContext.room_id` | id | n/a — becomes the `Rooms` instance's node id | Not a field under the node; it's the instance discriminator (`KnowledgeNode.id` in Phase 4). |
| `RoomContext.roomType` | scalar | `Project.Rooms.<instance>.RoomType` | |
| `RoomContext.budgetOrRequirement` | scalar | `Project.Rooms.<instance>.Budget` | |
| `RoomContext.style` | scalar | `Project.Rooms.<instance>.Style` | See **Gap 1** below — today's system also writes a duplicate freeform fact for this. |
| `RoomContext.squareFootage` | scalar | `Project.Rooms.<instance>.SquareFootage` | |
| `RoomContext.existingFurniture` | scalar | `Project.Rooms.<instance>.ExistingFurniture` | Free-text note distinct from the structured `Furniture` subtree, which holds newly-chosen furniture, not what's already in the room. |
| `RoomContext.materials[].item` | id | n/a — becomes the `Materials` instance's discriminator | e.g. `"flooring"`, `"countertop"`. |
| `RoomContext.materials[].material` | scalar | `Project.Rooms.<instance>.Materials.<item-instance>.Material` | |
| `RoomContext.materials[].specification` | scalar | `Project.Rooms.<instance>.Materials.<item-instance>.Specification` | |
| `field_status`, `field_attempts` | dict | **out of ontology** — `DialogueState` (Phase 7) | Retry/completeness bookkeeping, not project facts — already the master plan's own call for these two fields. |
| `ContextGraph` node type `project` (anchor) | node | `Project` (the root itself) | |
| `ContextGraph` node type `budget` (anchor) | node | `Project.Budget.Total` or `Project.Rooms.<instance>.Budget` | See **Gap 2** — collapses onto the same fact as the `PartialContext` budget rows above once migrated; today they're two independent representations. |
| `ContextGraph` node type `room` (anchor) | node | `Project.Rooms.<instance>` | Same instance as `RoomContext` above — collapses once migrated (`_reconcile_freeform_room_duplicates` already does this reconciliation at the `ContextGraph` level today). |
| `ContextGraph` node type `preference` | node | `Project.Requirements.ClientPreferences.<instance>` | See **Gap 1**. |
| `ContextGraph` node type `constraint` | node | `Project.Requirements.Constraints.<instance>` | |
| `ContextGraph` node type `attribute` (default/catch-all) | node | `Project.Rooms.<instance>.Attributes.<instance>` | See **Gap 3**. |
| `ContextGraph` node type `entity` | node | `Project.Rooms.<instance>.Furniture.<instance>` | See **Gap 3**. |
| `GraphRelation` `part_of`, `located_in` | edge | **not an ontology node** | Encodes containment that `canonical_path` itself expresses once nodes are migrated — see **Gap 4**. |
| `GraphRelation` `uses_material`, `applies_to`, `modifies`, `requires`, `budget_for`, `rejected_in_favor_of`, `revises` | edge | **not an ontology node** — carried forward as `KnowledgeEdge` relations (Phase 9) | Genuinely relational, not hierarchical — reused close to as-is. |

## 2. Instantiable nodes

`Rooms`, `Materials`, `Furniture`, `Attributes`, `Constraints`,
`ClientPreferences`, `Unmapped` (added post-Phase-5, see Gap 5 resolution
below), and `RoomLineItems` are `instantiable: true` — each can exist
zero-or-many times per project, keyed by an opaque instance id (mirroring
`RoomContext.room_id` today). Everything else (`Project`, `BasicInformation`,
`Budget`, `Timeline`, `Requirements`, `Quotation`) exists at most once per
project.

## 3. Versioning decision

**Additive-only**, per the plan's own recommendation: new node types and
fields may be added to a future `v2.yaml`; an existing `canonical_path` is
never renamed or repurposed, only deprecated (marked, not deleted, so
historical `KnowledgeNode` rows written against `v1` stay resolvable).
Recorded as `versioning_policy: additive-only` in `ontology/v1.yaml` itself
so it isn't just a convention living in this doc.

## 4. Known gaps — resolutions recorded as of Phase 5

These are real duplications/ambiguities in the **current** system, not
introduced by this ontology. Phase 3 flagged them without deciding; the
resolutions below were made while actually building Phase 5
(`app/canonical_mapper.py` — see `ontology/PHASE5_CANONICAL_MAPPER.md`),
since the mapper's own design forced each one to a concrete answer. A
**decision recorded** is not the same as **enacted in the live turn** —
none of these change `app/graph.py`'s behavior today; that's Phase 7's job
(the freeze rule in `ARCHITECTURE_BASELINE.md` still holds until then).

**Gap 1 — style duplication. RESOLVED (decision, not yet enacted):**
post-Phase-7, extraction produces one set of facts per turn (Phase 1/2's
`understand()`/`execute()` already replaced the old dual-branch routing —
see `ARCHITECTURE_BASELINE.md`), and *every* extracted fact, including a
style statement, is routed through `map_to_canonical()` to its one canonical
node. A style-like preference (`node_type_hint="Style"` or matched via
embedding against `Rooms`' description) resolves to
`Rooms.<instance>.Style`, not a parallel `ClientPreferences` node — there is
only ever one extraction pass and one mapping call per fact once Phase 7
retires today's separate `extract_fields`/`extract_graph_links` calls, so
the duplication is structurally impossible after cutover, not just guarded
against.

**Gap 2 — budget anchor duplication. RESOLVED (decision, not yet enacted):**
same resolution as Gap 1 — a budget figure is one fact, mapped once, to
`Budget.Total` or `Rooms.<instance>.Budget` depending on scope. No separate
anchor-node representation survives Phase 7.

**Gap 3 — attribute/entity catch-all. RESOLVED — this *is* Phase 5's
deliverable, not a blocker.** `map_to_canonical()` is exactly the
recognition logic this gap asked for: it embeds a raw entity mention against
every node type's `description` in `ontology/v1.yaml` (see `Materials` vs.
`Furniture` vs. `Attributes`' descriptions, written specifically to be
distinguishable) and picks the best-scoring type above threshold, instead of
defaulting everything to `attribute`/`entity`. A material-shaped mention
("quartz countertop") now scores higher against `Materials`' description
than `Furniture`'s and lands there. `Attributes` remains the catch-all only
for what still scores below threshold against every specific type.

**Gap 4 — `part_of`/`located_in` become largely redundant post-migration.
RESOLVED:** decided now rather than deferred to Phase 9 — once a fact has a
`canonical_path`, a separate `located_in`/`part_of` edge stating the same
containment is redundant and is **not** carried forward as a `KnowledgeEdge`
in Phase 9. The genuinely relational subset (`uses_material`, `applies_to`,
`modifies`, `requires`, `budget_for`, `rejected_in_favor_of`, `revises`)
still is.

**Gap 5 — freeform facts with no resolvable room. RESOLVED:** `ontology/v1.yaml`
gained a `Project.Unmapped` node (instantiable, additive change — no
existing `canonical_path` renamed, consistent with §3's versioning policy).
`map_to_canonical()` routes a fact here — flagged, never silently placed —
when either (a) no room context is available to scope a room-specific type,
or (b) nothing scores above the confidence threshold against any real node
type's description. `scripts/migrate_partial_context_to_tree.py` (Phase 4)
still doesn't migrate the freeform `ContextGraph` layer at all — that
limitation stands, since `ProjectContext` never captured it and most
historical sessions will already be past the 30-day TTL by the time a
migration runs. `Unmapped` matters for *live* mapping (Phase 7 onward), not
for backfilling history that no longer exists.

## 5. Sign-off checklist

- [x] Every `PartialContext` field mapped to a `canonical_path` or an
      explicit out-of-ontology designation (table above).
- [x] Every `ContextGraph` node type mapped (table above).
- [x] Every `GraphRelation` accounted for (table above — either superseded
      by path structure, or carried forward as a `KnowledgeEdge` relation).
- [x] Instantiable nodes identified and justified (§2).
- [x] Versioning policy decided and recorded in `ontology/v1.yaml` (§3).
- [x] Gaps 1–5 explicitly resolved (§4) — decisions recorded; enactment in
      the live turn is Phase 7's job, not retroactive to this doc.
- [ ] Reviewed and approved by whoever owns the quotation logic downstream
      (per the plan's own exit criteria — Phase 9's dependency graph and any
      future quote-generation work builds directly on this tree, especially
      the `Rooms.<instance>.Budget` / `Quotation` shape). Still open — no
      substitute for an actual human review, regardless of how many gaps
      got resolved along the way.
