from app.context_builder import ProposedWrite, apply_to_graph
from app.question_engine import find_knowledge_gaps

# Walk order (see PENDING_GAP_ANALYSIS.md / the pending_gap-centralization
# plan, updated for batched room questions): ProjectType -> Rooms existence
# -> one room's ENTIRE open-field set at once (active room stays in focus
# until it has nothing left open, then auto-advances to the next room with
# anything open, non-skipped rooms before skipped ones, oldest first) ->
# Timeline -> Budget.Total (moved here from position 2).


async def test_find_knowledge_gaps_starts_with_project_type_on_an_empty_project():
    batch = await find_knowledge_gaps("proj-gap-empty")
    assert len(batch.gaps) == 1
    assert batch.gaps[0].canonical_path == "Project.BasicInformation.ProjectType"
    assert batch.room_id is None


async def test_find_knowledge_gaps_asks_for_a_room_once_project_type_is_set():
    project_id = "proj-gap-no-room"
    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical")])

    batch = await find_knowledge_gaps(project_id)
    assert len(batch.gaps) == 1
    assert batch.gaps[0].canonical_path == "Project.Rooms.<new>"
    assert batch.gaps[0].node_type == "RoomType"


async def test_find_knowledge_gaps_batches_every_open_room_field_together():
    project_id = "proj-gap-room-fields"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
        ],
    )

    # RoomType is already filled — Budget/Style/SquareFootage/ExistingFurniture
    # all come back together in ONE batch, not one field per call.
    batch = await find_knowledge_gaps(project_id)
    assert batch.room_id == "r1"
    assert [g.node_type for g in batch.gaps] == ["Budget", "Style", "SquareFootage", "ExistingFurniture"]

    await apply_to_graph(project_id, [ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical")])
    batch2 = await find_knowledge_gaps(project_id)
    assert [g.node_type for g in batch2.gaps] == ["Style", "SquareFootage", "ExistingFurniture"]


async def test_find_knowledge_gaps_reaches_timeline_after_every_room_field_is_resolved():
    project_id = "proj-gap-timeline"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
            # Project.Budget.Total deliberately left unset — Timeline must be
            # asked before it now (Budget.Total moved to the very end).
        ],
    )

    batch = await find_knowledge_gaps(project_id)
    assert len(batch.gaps) == 1
    assert batch.gaps[0].canonical_path == "Project.Timeline.Value"
    assert batch.room_id is None


async def test_find_knowledge_gaps_checks_budget_total_last():
    project_id = "proj-gap-budget-last"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
            ProposedWrite(canonical_path="Project.Timeline.Value", node_type="Value", value="6 weeks", tier="moderate"),
        ],
    )

    batch = await find_knowledge_gaps(project_id)
    assert len(batch.gaps) == 1
    assert batch.gaps[0].canonical_path == "Project.Budget.Total"


async def test_find_knowledge_gaps_returns_none_once_everything_is_resolved():
    project_id = "proj-gap-complete"
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
            ProposedWrite(canonical_path="Project.Timeline.Value", node_type="Value", value="6 weeks", tier="moderate"),
            ProposedWrite(canonical_path="Project.Budget.Total", node_type="Total", value="$20k", tier="critical"),
        ],
    )

    assert await find_knowledge_gaps(project_id) is None


# ---------------------------------------------------------------------------
# Multi-room active-room focus + auto-advance + skip/revisit (see
# classify_intent_node's active-room derivation and decline-detection, which
# populates skipped_rooms).
# ---------------------------------------------------------------------------


async def _seed_two_rooms(project_id: str) -> None:
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.BasicInformation.ProjectType", node_type="ProjectType", value="renovation", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.RoomType", node_type="RoomType", value="kitchen", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r1.Budget", node_type="Budget", value="$8k", room_id="r1", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.RoomType", node_type="RoomType", value="bedroom", room_id="r2", tier="critical"),
            ProposedWrite(canonical_path="Project.Rooms.r2.Budget", node_type="Budget", value="$5k", room_id="r2", tier="critical"),
        ],
    )


async def test_find_knowledge_gaps_defaults_to_the_oldest_incomplete_room_with_no_active_room():
    project_id = "proj-gap-default-room"
    await _seed_two_rooms(project_id)

    batch = await find_knowledge_gaps(project_id)
    assert batch.room_id == "r1", "r1 was created first, so it's the default focus with no active_room_id"
    assert [g.node_type for g in batch.gaps] == ["Style", "SquareFootage", "ExistingFurniture"]


async def test_find_knowledge_gaps_stays_on_the_active_room_even_if_it_wasnt_created_first():
    project_id = "proj-gap-active-room-focus"
    await _seed_two_rooms(project_id)

    batch = await find_knowledge_gaps(project_id, active_room_id="r2")
    assert batch.room_id == "r2", "r2 is the active room and still has open fields, so it stays in focus"
    assert [g.node_type for g in batch.gaps] == ["Style", "SquareFootage", "ExistingFurniture"]


async def test_find_knowledge_gaps_auto_advances_once_the_active_room_is_fully_filled():
    project_id = "proj-gap-auto-advance"
    await _seed_two_rooms(project_id)
    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.r1.Style", node_type="Style", value="modern", room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.SquareFootage", node_type="SquareFootage", value=150, room_id="r1", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r1.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r1", tier="optional"),
        ],
    )

    # r1 (the active room) is now fully filled — auto-advance to r2, the
    # next room with anything open, without the caller having to re-mention it.
    batch = await find_knowledge_gaps(project_id, active_room_id="r1")
    assert batch.room_id == "r2"
    assert [g.node_type for g in batch.gaps] == ["Style", "SquareFootage", "ExistingFurniture"]


async def test_find_knowledge_gaps_revisits_a_skipped_room_only_after_every_other_room_is_exhausted():
    project_id = "proj-gap-skip-revisit"
    await _seed_two_rooms(project_id)

    batch = await find_knowledge_gaps(project_id, skipped_rooms=["r1"])
    assert batch.room_id == "r2", "r1 is skipped, so r2's own fields are asked first"

    await apply_to_graph(
        project_id,
        [
            ProposedWrite(canonical_path="Project.Rooms.r2.Style", node_type="Style", value="cozy", room_id="r2", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r2.SquareFootage", node_type="SquareFootage", value=120, room_id="r2", tier="moderate"),
            ProposedWrite(canonical_path="Project.Rooms.r2.ExistingFurniture", node_type="ExistingFurniture", value="none", room_id="r2", tier="optional"),
        ],
    )

    batch2 = await find_knowledge_gaps(project_id, active_room_id="r2", skipped_rooms=["r1"])
    assert batch2.room_id == "r1", (
        "once every non-skipped room (r2) is fully exhausted, the skipped room's own "
        "remaining fields are revisited — a room is never explicitly 'unskipped', it just "
        "becomes the only candidate left in the second pass"
    )
    assert [g.node_type for g in batch2.gaps] == ["Style", "SquareFootage", "ExistingFurniture"]
