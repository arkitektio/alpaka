import pytest
from alpaka.api.schema import (
    create_room,
    send,
    get_message,
    list_messages,
    search_messages,
    MessageFilter,
    MessageOrder,
    OffsetPaginationInput,
    Ordering,
    StructureInput,
)
from .conftest import DeployedAlpaka


@pytest.mark.integration
def test_send_message(deployed_app: DeployedAlpaka) -> None:
    """A message can be sent to a room by an agent."""
    room = create_room(title="Chat Room", description="A room to chat in.")

    message = send(
        text="Hello, world!",
        room=room.id,
        agent_id="test-agent",
    )
    assert message.typename == "Message"
    assert message.id is not None
    assert message.text == "Hello, world!"
    assert message.room.id == room.id
    assert message.agent.room.id == room.id


@pytest.mark.integration
def test_get_message(deployed_app: DeployedAlpaka) -> None:
    """A sent message can be fetched again by its id."""
    room = create_room(title="Fetch Message Room", description="Holds a message.")
    sent = send(text="Fetch this message", room=room.id, agent_id="test-agent")

    fetched = get_message(id=sent.id)
    assert fetched.id == sent.id
    assert fetched.text == "Fetch this message"
    assert fetched.room.id == room.id


@pytest.mark.integration
def test_messages_accumulate_in_room(deployed_app: DeployedAlpaka) -> None:
    """Subsequent messages see the earlier messages of the room as `before`."""
    room = create_room(title="History Room", description="Accumulates history.")

    send(text="First message", room=room.id, agent_id="test-agent")
    second = send(text="Second message", room=room.id, agent_id="test-agent")

    earlier_texts = {message.text for message in second.before}
    assert "First message" in earlier_texts


@pytest.mark.integration
def test_send_message_with_attached_structure(deployed_app: DeployedAlpaka) -> None:
    """A message can carry attached structures."""
    room = create_room(title="Structure Room", description="Holds structures.")

    message = send(
        text="Look at this structure",
        room=room.id,
        agent_id="test-agent",
        attach_structures=[
            StructureInput(identifier="@test/item", object="1"),
        ],
    )
    assert len(message.attached_structures) == 1
    structure = message.attached_structures[0]
    assert structure.identifier == "@test/item"
    assert structure.object == "1"


@pytest.mark.integration
def test_list_messages_contains_sent_message(deployed_app: DeployedAlpaka) -> None:
    """A sent message shows up when listing messages."""
    room = create_room(title="Listed Messages Room", description="Lists messages.")
    sent = send(text="List this message", room=room.id, agent_id="test-agent")

    messages = list_messages(pagination=OffsetPaginationInput(limit=100))
    assert sent.id in {message.id for message in messages}


@pytest.mark.integration
def test_filter_messages_by_id(deployed_app: DeployedAlpaka) -> None:
    """Messages can be filtered down to a specific id."""
    room = create_room(title="Filter Messages Room", description="Filter by id.")
    sent = send(text="Filter this message", room=room.id, agent_id="test-agent")

    messages = list_messages(filter=MessageFilter(ids=(sent.id,)))
    assert [message.id for message in messages] == [sent.id]


@pytest.mark.integration
def test_search_messages(deployed_app: DeployedAlpaka) -> None:
    """A message can be found through the search query."""
    room = create_room(title="Search Messages Room", description="Searchable.")
    sent = send(
        text="A uniquely searchable narwhal message",
        room=room.id,
        agent_id="test-agent",
    )

    options = search_messages(values=[sent.id])
    assert sent.id in {option.value for option in options}


@pytest.mark.integration
def test_order_messages_by_creation_descending(deployed_app: DeployedAlpaka) -> None:
    """Messages in a room can be ordered by creation time, newest first."""
    room = create_room(title="Ordered Messages Room", description="Ordered.")
    first = send(text="First ordered", room=room.id, agent_id="test-agent")
    second = send(text="Second ordered", room=room.id, agent_id="test-agent")
    third = send(text="Third ordered", room=room.id, agent_id="test-agent")
    ids = (first.id, second.id, third.id)

    messages = list_messages(
        filter=MessageFilter(ids=ids),
        order=[MessageOrder(createdAt=Ordering.DESC)],
    )
    assert [message.id for message in messages] == [third.id, second.id, first.id]


@pytest.mark.integration
def test_order_messages_by_creation_ascending(deployed_app: DeployedAlpaka) -> None:
    """Ordering messages ascending returns them oldest first."""
    room = create_room(title="Ascending Messages Room", description="Ascending.")
    first = send(text="Oldest message", room=room.id, agent_id="test-agent")
    second = send(text="Newest message", room=room.id, agent_id="test-agent")
    ids = (first.id, second.id)

    messages = list_messages(
        filter=MessageFilter(ids=ids),
        order=[MessageOrder(createdAt=Ordering.ASC)],
    )
    assert [message.id for message in messages] == [first.id, second.id]


@pytest.mark.integration
def test_search_messages_respects_limit(deployed_app: DeployedAlpaka) -> None:
    """The message search limit caps the number of returned options."""
    room = create_room(title="Limited Search Room", description="Limited.")
    for index in range(3):
        send(
            text=f"Limitword message {index}",
            room=room.id,
            agent_id="test-agent",
        )

    options = search_messages(search="Limitword", limit=2)
    assert len(options) <= 2


@pytest.mark.integration
def test_search_messages_offset_paginates(deployed_app: DeployedAlpaka) -> None:
    """Offset skips earlier message search results so pages do not overlap."""
    room = create_room(title="Paged Search Room", description="Paged.")
    for index in range(4):
        send(
            text=f"Offsetword message {index}",
            room=room.id,
            agent_id="test-agent",
        )

    first_page = search_messages(search="Offsetword", limit=2, offset=0)
    second_page = search_messages(search="Offsetword", limit=2, offset=2)

    first_ids = {option.value for option in first_page}
    second_ids = {option.value for option in second_page}
    assert first_ids.isdisjoint(second_ids)
