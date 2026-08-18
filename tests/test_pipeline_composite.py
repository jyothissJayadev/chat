"""End-to-end composite-entity write flow — proves the bedcover scenario
the user reported ("add a bedcover to the bed in the bedroom") now lands
the bedcover as a Parts child of the bed, NOT as a sibling under the
room's Furniture container.

Builds the live graph (a bedroom with a bed already in it) the way the
real pipeline does, then calls app.pipeline._apply_update with the exact
ContextChangeResult shape app.llm.resolve_context_changes would return
under the new composite-entity prompt contract (rule 9): the bed is the
parent mention (existing_path = the bed's path), and the bedcover is a
`parts` entry on it. No LLM call — the resolver result is constructed by
hand, so this tests the write path (split_connection -> _apply_update ->
map_part_to_canonical) in isolation.
"""

from unittest.mock import patch

import pytest

from app import canonical_mapper, context_builder, graph_store, llm, pipeline
from app.context_builder import ProposedWrite
from app.tasks import TaskSpec, TaskType

ROOM_ID = "bedroom01"
PROJECT_ID = "proj-composite-e2e"


async def _seed_bedroom_with_bed() -> str:
    """Create Project.Rooms.<ROOM_ID> with RoomType="Bedroom" and a
    Furniture.beds instance (Label="beds") — the exact tree state from the
    user's reported scenario. Returns the bed's full canonical path."""
    await context_builder.apply_to_graph(
        PROJECT_ID,
        [
            ProposedWrite(
                canonical_path=f"Project.Rooms.{ROOM_ID}.RoomType",
                node_type="RoomType",
                value="Bedroom",
                room_id=ROOM_ID,
            ),
            ProposedWrite(
                canonical_path=f"Project.Rooms.{ROOM_ID}.Furniture.beds.Label",
                node_type="Label",
                value="beds",
                room_id=ROOM_ID,
            ),
        ],
    )
    return f"Project.Rooms.{ROOM_ID}.Furniture.beds"


def _bedcover_result(bed_root_relative_path: str) -> llm.ContextChangeResult:
    """The ContextChangeResult resolve_context_changes should return for
    "add a bedcover to the bed in the bedroom" under the new contract: the
    bed is the parent mention (existing_path = bed), the bedcover is a part
    on it."""
    return llm.ContextChangeResult(
        text="add a bedcover to the bed in the bedroom",
        intent="CONTEXT_UPDATE",
        fields=llm.ContextChangeFields(),
        freeform_entities=[
            llm.FreeformEntityChange(
                raw_entity="beds",
                existing_path=bed_root_relative_path,
                parts=[
                    llm.PartChange(raw_entity="bedcover"),
                ],
            )
        ],
    )


async def test_bedcover_lands_as_a_part_of_the_bed_not_a_sibling():
    bed_path = await _seed_bedroom_with_bed()
    bed_root_relative = f"Rooms.{ROOM_ID}.Furniture.beds"

    result = _bedcover_result(bed_root_relative)
    changes: list[dict] = []
    written: list[str] = []
    with patch("app.canonical_mapper.llm.embed", side_effect=_zero_embed):
        await pipeline._apply_update(
            PROJECT_ID, ROOM_ID, result, [], changes, written,
            parent_instance_path=bed_path,
        )

    # The bedcover must exist as a Parts child of the bed, with a Label.
    bedcover_path = f"{bed_path}.Parts.bedcover"
    bedcover_instance = await graph_store.find_one(PROJECT_ID, bedcover_path)
    assert bedcover_instance is not None, "bedcover part must be created under the bed"
    assert bedcover_instance.node_type == "Parts"
    label = await graph_store.find_one(PROJECT_ID, f"{bedcover_path}.Label")
    assert label is not None and label.value == "bedcover"

    # And it must NOT be a sibling Furniture instance under the room — the
    # exact bug the user reported.
    sibling = await graph_store.find_one(PROJECT_ID, f"Project.Rooms.{ROOM_ID}.Furniture.bedcover")
    assert sibling is None, "bedcover must not be created as a sibling under the room's Furniture container"


async def test_red_sofa_with_cover_and_pillows_writes_full_composite_tree():
    """The richer composite case: a NEW sofa with a color property, a velvet
    cover part, and two pillows part — connection is room-level, so parts
    attach under the sofa's own just-created instance."""
    result = llm.ContextChangeResult(
        text="add a red sofa with the velvet cover and two pillows on top of this",
        intent="CONTEXT_UPDATE",
        fields=llm.ContextChangeFields(),
        freeform_entities=[
            llm.FreeformEntityChange(
                raw_entity="sofa",
                node_type_hint="Furniture",
                properties=[llm.PropertyChange(name="color", value="red")],
                parts=[
                    llm.PartChange(raw_entity="velvet cover", material="velvet"),
                    llm.PartChange(raw_entity="pillows", quantity="2"),
                ],
            )
        ],
    )
    changes: list[dict] = []
    written: list[str] = []
    with patch("app.canonical_mapper.llm.embed", side_effect=_zero_embed):
        await pipeline._apply_update(PROJECT_ID, ROOM_ID, result, [], changes, written)

    sofa_path = f"Project.Rooms.{ROOM_ID}.Furniture.sofa"
    assert (await graph_store.find_one(PROJECT_ID, sofa_path)) is not None

    # color property
    color_path = f"{sofa_path}.Properties.color"
    assert (await graph_store.find_one(PROJECT_ID, color_path)) is not None
    color_value = await graph_store.find_one(PROJECT_ID, f"{color_path}.Value")
    assert color_value is not None and color_value.value == "red"

    # velvet cover part + its Material leaf
    cover_path = f"{sofa_path}.Parts.velvet_cover"
    assert (await graph_store.find_one(PROJECT_ID, cover_path)) is not None
    cover_material = await graph_store.find_one(PROJECT_ID, f"{cover_path}.Material")
    assert cover_material is not None and cover_material.value == "velvet"

    # pillows part + its Quantity leaf
    pillows_path = f"{sofa_path}.Parts.pillows"
    assert (await graph_store.find_one(PROJECT_ID, pillows_path)) is not None
    pillows_qty = await graph_store.find_one(PROJECT_ID, f"{pillows_path}.Quantity")
    assert pillows_qty is not None and pillows_qty.value == "2"


async def test_render_tree_text_shows_nested_parts_and_properties():
    """render_project_tree_text must surface the composite children so the
    next turn's resolve_context_changes can ground against them — the
    renderer walks Parts/Properties recursively (context_builder change)."""
    project_id = "proj-composite-render"
    # Create the sofa via map_to_canonical (NOT apply_to_graph) so it gets
    # node_type="Furniture" and surfaces in the room's instance scan — the
    # same path the live pipeline takes for a new freeform entity.
    with patch("app.canonical_mapper.llm.embed", side_effect=_zero_embed):
        await canonical_mapper.map_to_canonical("sofa", "Furniture", project_id, room_id=ROOM_ID)
        sofa_path = f"Project.Rooms.{ROOM_ID}.Furniture.sofa"
        await canonical_mapper.map_property_to_canonical("color", "red", sofa_path, project_id, ROOM_ID)
        await canonical_mapper.map_part_to_canonical("pillows", sofa_path, project_id, ROOM_ID)
        await context_builder.apply_to_graph(
            project_id,
            [ProposedWrite(canonical_path=f"{sofa_path}.Parts.pillows.Quantity", node_type="Quantity", value="2", room_id=ROOM_ID)],
        )

    tree = await context_builder.render_project_tree_text(project_id)
    assert "Furniture.sofa" in tree
    assert "Properties.color" in tree, "the color property must render nested under the sofa"
    assert "Parts.pillows" in tree, "the pillows part must render nested under the sofa"
    assert 'Quantity="2"' in tree


async def _zero_embed(texts: list[str]) -> list[list[float]]:
    """A zero vector for any text: cosine similarity collapses to 0, so the
    embedding-dedup path never merges two distinct parts/properties (each
    creates its own instance). Used where sibling candidates exist and the
    mapper legitimately needs to score a query against them — the real
    pipeline would call the live embed model here."""
    return [[0.0] * 8 for _ in texts]
