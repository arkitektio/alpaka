import pytest

from alpaka.alpaka import Alpaka
from alpaka.api.schema import (
    MessageFilter,
    MessageOrder,
    OffsetPaginationInput,
    Ordering,
    StructureInput,
)


@pytest.mark.integration
def test_send_message(alpaka: Alpaka) -> None:
    """A message can be sent to a room by an agent."""
    room = alpaka.create_room(title="Chat Room", description="A room to chat in.")

    message = alpaka.send(
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
def test_get_message(alpaka: Alpaka) -> None:
    """A sent message can be fetched again by its id."""
    room = alpaka.create_room(title="Fetch Message Room", description="Holds a message.")
    sent = alpaka.send(text="Fetch this message", room=room.id, agent_id="test-agent")

    fetched = alpaka.get_message(id=sent.id)
    assert fetched.id == sent.id
    assert fetched.text == "Fetch this message"
    assert fetched.room.id == room.id


@pytest.mark.integration
def test_messages_accumulate_in_room(alpaka: Alpaka) -> None:
    """Subsequent messages see the earlier messages of the room as `before`."""
    room = alpaka.create_room(title="History Room", description="Accumulates history.")

    alpaka.send(text="First message", room=room.id, agent_id="test-agent")
    second = alpaka.send(text="Second message", room=room.id, agent_id="test-agent")

    earlier_texts = {message.text for message in second.before}
    assert "First message" in earlier_texts


@pytest.mark.integration
def test_send_message_with_attached_structure(alpaka: Alpaka) -> None:
    """A message can carry attached structures."""
    room = alpaka.create_room(title="Structure Room", description="Holds structures.")

    message = alpaka.send(
        text="Look at this structure",
        room=room.id,
        agent_id="test-agent",
        attach_structures=[
            StructureInput(identifier="@test/item", object=1),
        ],
    )
    assert len(message.attached_structures) == 1
    structure = message.attached_structures[0]
    assert structure.identifier == "@test/item"
    assert structure.object == 1


@pytest.mark.integration
def test_list_messages_contains_sent_message(alpaka: Alpaka) -> None:
    """A sent message shows up when listing messages."""
    room = alpaka.create_room(title="Listed Messages Room", description="Lists messages.")
    sent = alpaka.send(text="List this message", room=room.id, agent_id="test-agent")

    messages = alpaka.list_messages(pagination=OffsetPaginationInput(limit=100))
    assert sent.id in {message.id for message in messages}


@pytest.mark.integration
def test_filter_messages_by_id(alpaka: Alpaka) -> None:
    """Messages can be filtered down to a specific id."""
    room = alpaka.create_room(title="Filter Messages Room", description="Filter by id.")
    sent = alpaka.send(text="Filter this message", room=room.id, agent_id="test-agent")

    messages = alpaka.list_messages(filter=MessageFilter(ids=(sent.id,)))
    assert [message.id for message in messages] == [sent.id]


@pytest.mark.integration
def test_search_messages(alpaka: Alpaka) -> None:
    """A message can be found through the search query."""
    room = alpaka.create_room(title="Search Messages Room", description="Searchable.")
    sent = alpaka.send(
        text="A uniquely searchable narwhal message",
        room=room.id,
        agent_id="test-agent",
    )

    options = alpaka.search_messages(values=[sent.id])
    assert sent.id in {option.value for option in options}


@pytest.mark.integration
def test_order_messages_by_creation_descending(alpaka: Alpaka) -> None:
    """Messages in a room can be ordered by creation time, newest first."""
    room = alpaka.create_room(title="Ordered Messages Room", description="Ordered.")
    first = alpaka.send(text="First ordered", room=room.id, agent_id="test-agent")
    second = alpaka.send(text="Second ordered", room=room.id, agent_id="test-agent")
    third = alpaka.send(text="Third ordered", room=room.id, agent_id="test-agent")
    ids = (first.id, second.id, third.id)

    messages = alpaka.list_messages(
        filter=MessageFilter(ids=ids),
        order=[MessageOrder(createdAt=Ordering.DESC)],
    )
    assert [message.id for message in messages] == [third.id, second.id, first.id]


@pytest.mark.integration
def test_order_messages_by_creation_ascending(alpaka: Alpaka) -> None:
    """Ordering messages ascending returns them oldest first."""
    room = alpaka.create_room(title="Ascending Messages Room", description="Ascending.")
    first = alpaka.send(text="Oldest message", room=room.id, agent_id="test-agent")
    second = alpaka.send(text="Newest message", room=room.id, agent_id="test-agent")
    ids = (first.id, second.id)

    messages = alpaka.list_messages(
        filter=MessageFilter(ids=ids),
        order=[MessageOrder(createdAt=Ordering.ASC)],
    )
    assert [message.id for message in messages] == [first.id, second.id]


@pytest.mark.integration
def test_search_messages_respects_limit(alpaka: Alpaka) -> None:
    """The message search limit caps the number of returned options."""
    room = alpaka.create_room(title="Limited Search Room", description="Limited.")
    for index in range(3):
        alpaka.send(
            text=f"Limitword message {index}",
            room=room.id,
            agent_id="test-agent",
        )

    options = alpaka.search_messages(search="Limitword", limit=2)
    assert len(options) <= 2


@pytest.mark.integration
def test_search_messages_offset_paginates(alpaka: Alpaka) -> None:
    """Offset skips earlier message search results so pages do not overlap."""
    room = alpaka.create_room(title="Paged Search Room", description="Paged.")
    for index in range(4):
        alpaka.send(
            text=f"Offsetword message {index}",
            room=room.id,
            agent_id="test-agent",
        )

    first_page = alpaka.search_messages(search="Offsetword", limit=2, offset=0)
    second_page = alpaka.search_messages(search="Offsetword", limit=2, offset=2)

    first_ids = {option.value for option in first_page}
    second_ids = {option.value for option in second_page}
    assert first_ids.isdisjoint(second_ids)
