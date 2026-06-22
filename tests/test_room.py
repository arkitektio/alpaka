import pytest
from alpaka.api.schema import (
    create_room,
    get_room,
    list_rooms,
    search_rooms,
    OffsetPaginationInput,
    Ordering,
    RoomFilter,
    RoomOrder,
)
from .conftest import DeployedAlpaka


@pytest.mark.integration
def test_create_room(deployed_app: DeployedAlpaka) -> None:
    """A room can be created with a title and description."""
    room = create_room(
        title="Test Room",
        description="This is a test room.",
    )
    assert room.typename == "Room"
    assert room.id is not None
    assert room.title == "Test Room"
    assert room.description == "This is a test room."


@pytest.mark.integration
def test_create_room_without_arguments(deployed_app: DeployedAlpaka) -> None:
    """A room can be created without an explicit title or description."""
    room = create_room()
    assert room.typename == "Room"
    assert room.id is not None


@pytest.mark.integration
def test_get_room(deployed_app: DeployedAlpaka) -> None:
    """A created room can be fetched again by its id."""
    created = create_room(title="Fetch Me", description="A room to fetch.")

    fetched = get_room(id=created.id)
    assert fetched.id == created.id
    assert fetched.title == "Fetch Me"
    assert fetched.description == "A room to fetch."


@pytest.mark.integration
def test_list_rooms_contains_created_room(deployed_app: DeployedAlpaka) -> None:
    """A created room shows up when listing rooms."""
    created = create_room(title="Listed Room", description="Should be listed.")

    rooms = list_rooms(pagination=OffsetPaginationInput(limit=100))
    assert created.id in {room.id for room in rooms}


@pytest.mark.integration
def test_search_rooms_by_title(deployed_app: DeployedAlpaka) -> None:
    """A room can be found through the search query by its title."""
    created = create_room(
        title="Searchable Unicorn Room",
        description="A uniquely titled room.",
    )

    options = search_rooms(search="Unicorn")
    assert created.id in {option.value for option in options}


@pytest.mark.integration
def test_filter_rooms_by_id(deployed_app: DeployedAlpaka) -> None:
    """Rooms can be filtered down to a specific id."""
    created = create_room(title="Filtered Room", description="Filter by id.")

    rooms = list_rooms(filter=RoomFilter(ids=(created.id,)))
    assert [room.id for room in rooms] == [created.id]


@pytest.mark.integration
def test_order_rooms_by_creation_ascending(deployed_app: DeployedAlpaka) -> None:
    """Rooms can be ordered by their creation time, oldest first."""
    first = create_room(title="Order A", description="First created.")
    second = create_room(title="Order B", description="Second created.")
    third = create_room(title="Order C", description="Third created.")
    ids = (first.id, second.id, third.id)

    rooms = list_rooms(
        filter=RoomFilter(ids=ids),
        order=[RoomOrder(createdAt=Ordering.ASC)],
    )
    assert [room.id for room in rooms] == list(ids)


@pytest.mark.integration
def test_order_rooms_by_creation_descending(deployed_app: DeployedAlpaka) -> None:
    """Ordering by creation time descending reverses the result."""
    first = create_room(title="Desc A", description="First created.")
    second = create_room(title="Desc B", description="Second created.")
    third = create_room(title="Desc C", description="Third created.")
    ids = (first.id, second.id, third.id)

    rooms = list_rooms(
        filter=RoomFilter(ids=ids),
        order=[RoomOrder(createdAt=Ordering.DESC)],
    )
    assert [room.id for room in rooms] == [third.id, second.id, first.id]


@pytest.mark.integration
def test_order_rooms_by_title(deployed_app: DeployedAlpaka) -> None:
    """Rooms can be ordered alphabetically by title."""
    gamma = create_room(title="Zeta Title Room", description="Sorts last.")
    alpha = create_room(title="Alpha Title Room", description="Sorts first.")
    beta = create_room(title="Mu Title Room", description="Sorts middle.")
    ids = (gamma.id, alpha.id, beta.id)

    rooms = list_rooms(
        filter=RoomFilter(ids=ids),
        order=[RoomOrder(title=Ordering.ASC)],
    )
    assert [room.id for room in rooms] == [alpha.id, beta.id, gamma.id]


@pytest.mark.integration
def test_search_rooms_respects_limit(deployed_app: DeployedAlpaka) -> None:
    """The search limit caps the number of returned options."""
    for index in range(3):
        create_room(title=f"Limitsearch Room {index}", description="Limited search.")

    options = search_rooms(search="Limitsearch", limit=2)
    assert len(options) <= 2


@pytest.mark.integration
def test_search_rooms_offset_paginates(deployed_app: DeployedAlpaka) -> None:
    """Offset skips earlier search results so pages do not overlap."""
    for index in range(4):
        create_room(title=f"Offsetsearch Room {index}", description="Paged search.")

    first_page = search_rooms(search="Offsetsearch", limit=2, offset=0)
    second_page = search_rooms(search="Offsetsearch", limit=2, offset=2)

    first_ids = {option.value for option in first_page}
    second_ids = {option.value for option in second_page}
    assert first_ids.isdisjoint(second_ids)
