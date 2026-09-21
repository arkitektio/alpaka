"""The room-streaming helper, proved at the wire.

No docker: an ``AsyncMockLink`` terminates the rath chain, so these assert
exactly which mutations ``stream_into_room`` sends and with what -- the coalescing,
the ordering, and the two guarantees the server's README asks every client for
(finish always runs, and it carries the full text so a lost delta is repaired).
"""

from typing import Any

import pytest
from rath.links.testing.mock import AsyncMockLink
from rath.operation import Operation

from alpaka.alpaka import Alpaka
from alpaka.rath import AlpakaRath
from alpaka.streaming import stream_into_room

from .conftest import StaticTokens


def client(rath: AlpakaRath) -> Alpaka:
    """An Alpaka over the given rath, with a static tunnel endpoint."""
    return Alpaka(rath=rath, llm_url="http://testserver/llm/v1", tokens=StaticTokens())


def message_payload(message_id: str = "1", text: str = "", is_streaming: bool = True) -> dict:
    return {
        "id": message_id,
        "text": text,
        "isStreaming": is_streaming,
        "agent": {"id": "7"},
        "attachedStructures": [],
    }


def recording_rath() -> tuple[AlpakaRath, list[tuple[str, dict]]]:
    """A rath that records every streaming mutation it is asked to send."""
    calls: list[tuple[str, dict]] = []
    accumulated = {"text": ""}

    async def start(operation: Operation) -> dict:
        calls.append(("startMessage", dict(operation.variables)))
        return message_payload()

    async def append(operation: Operation) -> dict:
        calls.append(("appendMessage", dict(operation.variables)))
        accumulated["text"] += operation.variables["delta"]
        return message_payload(text=accumulated["text"])

    async def finish(operation: Operation) -> dict:
        calls.append(("finishMessage", dict(operation.variables)))
        return message_payload(text=operation.variables.get("text") or accumulated["text"], is_streaming=False)

    link = AsyncMockLink(mutation_resolver={"startMessage": start, "appendMessage": append, "finishMessage": finish})
    return AlpakaRath(link=link), calls


async def test_deltas_are_coalesced_and_the_message_is_finished_with_the_full_text() -> None:
    rath, calls = recording_rath()
    async with rath:
        async with stream_into_room(client(rath), room="3", agent_id="assistant", flush_chars=10, flush_interval=999) as reply:
            for token in ("Hel", "lo ", "wor", "ld!"):
                await reply.append(token)

    kinds = [name for name, _ in calls]
    assert kinds[0] == "startMessage" and kinds[-1] == "finishMessage"
    # Twelve characters over four tokens, flushed at ten: one append, not four.
    assert kinds.count("appendMessage") == 1
    assert calls[1][1]["delta"] == "Hello world!"
    # The finish carries the whole body, so a delta lost on the way is repaired.
    assert calls[-1][1]["text"] == "Hello world!"


async def test_a_raising_body_still_finishes_the_message() -> None:
    """The failure the server cannot recover from is a message left streaming."""
    rath, calls = recording_rath()
    with pytest.raises(RuntimeError):
        async with rath:
            async with stream_into_room(client(rath), room="3", agent_id="assistant", flush_chars=999, flush_interval=999) as reply:
                await reply.append("half a th")
                raise RuntimeError("the model died")

    kinds = [name for name, _ in calls]
    assert kinds[-1] == "finishMessage"
    # Buffered text is flushed before the finish rather than thrown away.
    assert "appendMessage" in kinds
    assert calls[-1][1]["text"] == "half a th"


async def test_camel_case_and_unset_omission_survive_regeneration() -> None:
    rath, calls = recording_rath()
    async with rath:
        async with stream_into_room(client(rath), room="3", agent_id="assistant") as reply:
            await reply.append("hi")

    start_variables = calls[0][1]
    assert start_variables["agentId"] == "assistant"
    assert start_variables["room"] == "3"
    assert "parent" not in start_variables


async def test_attached_structures_ride_along_with_the_finish() -> None:
    """``attachStructures`` is a field of ``finishMessage``, so a structure produced
    while the answer is still being written lands with it in one write."""
    from alpaka.api.schema import StructureInput

    rath, calls = recording_rath()
    async with rath:
        async with stream_into_room(client(rath), room="3", agent_id="assistant") as reply:
            await reply.append("here it is")
            reply.attach(StructureInput(identifier="@mikro/arraydataset", object="12"))

    finish = calls[-1][1]
    # object is Int! in the schema, so the id coerces on the way out.
    assert finish["attachStructures"] == [{"identifier": "@mikro/arraydataset", "object": 12}]


async def test_nothing_attached_means_the_field_is_omitted() -> None:
    rath, calls = recording_rath()
    async with rath:
        async with stream_into_room(client(rath), room="3", agent_id="assistant") as reply:
            await reply.append("plain")

    assert "attachStructures" not in calls[-1][1]
