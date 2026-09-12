"""Streaming a reply into a room, with the batching the server asks for.

Agents live on clients: the server never calls an LLM on behalf of a room. A
client that has a stream of tokens -- from :func:`alpaka.openai`, from the
``chat`` mutation, from anywhere -- forwards them into a room with
``startMessage`` / ``appendMessage`` / ``finishMessage``.

Doing that by hand goes wrong in the same three ways every time: one mutation
per token, a delta that arrives out of order because the previous call was not
awaited, and a message left ``isStreaming`` forever because the generator raised.
:func:`stream_into_room` is that discipline written once::

    async with stream_into_room(room=room.id, agent_id="assistant") as reply:
        async for chunk in completion:
            await reply.append(chunk.choices[0].delta.content or "")

The message is finished on the way out, including when the body raises, and the
full text is sent with the finish so a delta lost on the way is repaired.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from alpaka.api.schema import UNSET, ListMessage, StructureInput, aappend_message, afinish_message, astart_message

#: Deltas smaller than this wait for a sibling, unless the interval expires first.
FLUSH_CHARS = 30
#: Longest a delta waits before it is sent anyway.
FLUSH_INTERVAL_SECONDS = 0.15


class RoomStream:
    """An open streaming message. Do not construct directly; see :func:`stream_into_room`."""

    def __init__(self, message: ListMessage, *, flush_chars: int, flush_interval: float, rath: Any = None) -> None:
        self.message = message
        self.text = message.text or ""
        self._buffer: list[str] = []
        self._buffered_chars = 0
        self._flush_chars = flush_chars
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()
        self._rath = rath
        self._structures: list[StructureInput] = []
        # Appends must arrive in order, and the server concatenates them as they
        # land: one lock means a caller that forgets to await cannot interleave.
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        """The streaming message's id."""
        return self.message.id

    def attach(self, *structures: StructureInput) -> None:
        """Attach structures to the finished message.

        Collected rather than sent: ``attachStructures`` is a field of
        ``finishMessage``, so anything produced while the answer is still being
        written lands with it, in one write, when the stream closes.
        """
        self._structures.extend(structures)

    @property
    def structures(self) -> list[StructureInput]:
        """The structures that will be attached when this stream finishes."""
        return list(self._structures)

    async def append(self, delta: str) -> None:
        """Add a delta, sending it once it is big enough or old enough."""
        if not delta:
            return
        self._buffer.append(delta)
        self._buffered_chars += len(delta)
        if self._buffered_chars >= self._flush_chars or (time.monotonic() - self._last_flush) >= self._flush_interval:
            await self.flush()

    async def flush(self) -> None:
        """Send whatever is buffered now."""
        async with self._lock:
            if not self._buffer:
                return
            delta = "".join(self._buffer)
            self._buffer.clear()
            self._buffered_chars = 0
            self._last_flush = time.monotonic()
            self.text += delta
            await aappend_message(message=self.message.id, delta=delta, rath=self._rath)


@asynccontextmanager
async def stream_into_room(
    *,
    room: Any,
    agent_id: str,
    parent: Optional[Any] = None,
    text: str = "",
    flush_chars: int = FLUSH_CHARS,
    flush_interval: float = FLUSH_INTERVAL_SECONDS,
    rath: Any = None,
) -> AsyncIterator[RoomStream]:
    """Open a streaming message, yield a writer for it, and always finish it.

    Anything handed to :meth:`RoomStream.attach` rides along with the finish.

    The finish carries the full accumulated text, which is what makes a dropped
    delta recoverable: the server replaces the body rather than trusting the
    concatenation. It runs in a ``finally``, so a raising body still closes the
    message instead of leaving it streaming forever.
    """
    # UNSET rather than None: an omitted optional stays omitted on the wire.
    message = await astart_message(room=room, agent_id=agent_id, parent=UNSET if parent is None else parent, text=text, rath=rath)
    stream = RoomStream(message, flush_chars=flush_chars, flush_interval=flush_interval, rath=rath)
    try:
        yield stream
    finally:
        try:
            await stream.flush()
        finally:
            stream.message = await afinish_message(
                message=message.id,
                text=stream.text,
                attach_structures=stream.structures or UNSET,
                rath=rath,
            )
