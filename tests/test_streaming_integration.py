"""Streaming a reply into a room, against a real server.

The unit layer (``tests/test_stream_into_room.py``) proves what the helper puts
on the wire; this proves the server agrees -- that deltas concatenate, that the
message closes, and that a subscriber sees the three ``MESSAGE_*`` events. The
room subscription had no test at any level before this.

Everything here is the synchronous surface, driven the way a sync consumer would:
the ``deployed_app`` fixture enters the ``Alpaka`` composition synchronously, so
the subscription runs in a thread rather than a second event loop.
"""

import contextvars
import queue
import threading

import pytest

from alpaka.alpaka import Alpaka

from alpaka.api.schema import (
    RoomEventKind,
)



@pytest.mark.integration
def test_start_append_finish_accumulates(alpaka: Alpaka) -> None:
    """The server concatenates the deltas; finishing closes the message."""
    room = alpaka.create_room(title="Streaming Room", description="Tokens land here.")

    message = alpaka.start_message(room=room.id, agent_id="assistant")
    assert message.is_streaming is True

    for token in ("Hel", "lo ", "world"):
        alpaka.append_message(message=message.id, delta=token)

    finished = alpaka.finish_message(message=message.id, text="Hello world")
    assert finished.is_streaming is False

    assert alpaka.get_message(id=message.id).text == "Hello world"


@pytest.mark.integration
def test_a_subscriber_sees_created_updated_finished(alpaka: Alpaka) -> None:
    """The three ``MESSAGE_*`` events, in order, with the text accumulating."""
    room = alpaka.create_room(title="Watched Room", description="Has a listener.")
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def listen() -> None:
        try:
            for event in alpaka.watch_room(room=room.id, agent_id="listener"):
                events.put(event)
                if stop.is_set():
                    return
        except Exception as e:
            # The websocket is torn down with the deployment; an abandoned
            # subscription raising SubscriptionDisconnect on the way out is not a
            # test failure, but an unhandled *thread* exception would be.
            events.put(e)

    # copy_context, because the koil loop the composition runs on lives in a
    # contextvar and a bare thread starts with an empty context ("No koil context
    # found"). This is what any sync consumer subscribing off the main thread needs.
    listener = threading.Thread(target=contextvars.copy_context().run, args=(listen,), daemon=True)
    listener.start()

    # The subscription must be registered with the room's group before the first
    # broadcast: events are not backfilled, so anything sent while it is still
    # joining is simply never reported. Wait for that by round-tripping a
    # throwaway message rather than sleeping a hopeful number of seconds.
    for attempt in range(10):
        warmup = alpaka.start_message(room=room.id, agent_id="warmup")
        try:
            first = events.get(timeout=3)
        except queue.Empty:
            alpaka.finish_message(message=warmup.id)
            continue
        if isinstance(first, Exception):
            raise AssertionError(f"the subscription failed: {first!r}")
        alpaka.finish_message(message=warmup.id)
        break
    else:  # pragma: no cover - the subscription never joined
        raise AssertionError("the subscription never started receiving events")

    # Earlier warmup attempts land once the subscription is live; drain until the
    # room goes quiet so the assertions below see only their own events.
    while True:
        try:
            events.get(timeout=2)
        except queue.Empty:
            break

    message = alpaka.start_message(room=room.id, agent_id="assistant")
    created = events.get(timeout=15)
    alpaka.append_message(message=message.id, delta="hi")
    updated = events.get(timeout=15)
    alpaka.finish_message(message=message.id, text="hi")
    finished = events.get(timeout=15)
    stop.set()

    assert created.kind == RoomEventKind.MESSAGE_CREATED
    assert updated.kind == RoomEventKind.MESSAGE_UPDATED and updated.message.text == "hi"
    assert finished.kind == RoomEventKind.MESSAGE_FINISHED
    assert finished.message.is_streaming is False
