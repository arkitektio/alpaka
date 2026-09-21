"""An alpaka call goes through the client it is made on, and nothing else.

No server: the rath is a fake returning canned data, or an ``AsyncMockLink``.
"""

from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

import pytest
from pydantic import BaseModel, ConfigDict
from rath.links.testing.mock import AsyncMockLink
from rath.operation import Operation
from rath.origin import ContextBound, get_origin

from alpaka.alpaka import TASK_HEADER, Alpaka
from alpaka.rath import AlpakaRath
from alpaka.streaming import stream_into_room

from .conftest import StaticTokens

LLM_URL = "http://testserver/llm/v1"


class FakeRath:
    """Answers every query with the same canned room, and remembers what was sent."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.headers: list[Any] = []

    def _answer(self, variables: dict[str, Any]) -> Any:
        self.sent.append(variables)
        return SimpleNamespace(data={"room": {"id": "room-1", "agent": {"id": "a-1"}}})

    async def aquery(
        self, document: str, variables: dict[str, Any], headers: Any = None
    ) -> Any:
        self.headers.append(headers)
        return self._answer(variables)

    async def asubscribe(
        self, document: str, variables: dict[str, Any], headers: Any = None
    ) -> AsyncIterator[Any]:
        self.headers.append(headers)
        yield self._answer(variables)


class Agent(ContextBound):
    model_config = ConfigDict(frozen=True)
    id: str


class Room(ContextBound):
    model_config = ConfigDict(frozen=True)
    id: str
    agent: Agent


class GetRoom(BaseModel):
    """Shaped like a generated operation."""

    room: Room

    class Arguments(BaseModel):
        id: str
        note: Optional[str] = None

    class Meta:
        document = "query GetRoom($id: ID!) { room(id: $id) { id agent { id } } }"


def client(rath: AlpakaRath | None = None) -> Alpaka:
    """The real client, with a static tunnel endpoint: over ``rath``, or over a
    fake rath built without validation."""
    if rath is None:
        return Alpaka.model_construct(
            rath=FakeRath(), task_token=None, llm_url=LLM_URL, tokens=StaticTokens()
        )
    return Alpaka(rath=rath, llm_url=LLM_URL, tokens=StaticTokens())


async def test_aexecute_goes_through_the_client_and_results_remember_it() -> None:
    mine, other = client(), client()

    result = await mine.aexecute(GetRoom, {"id": "room-1"})

    assert (len(mine.rath.sent), len(other.rath.sent)) == (1, 0)
    origin = get_origin(result.room.agent)
    assert origin is not None and origin.client is mine and origin.rath is mine.rath


def test_execute_goes_through_the_client() -> None:
    from koil import Koil

    mine = client()
    with Koil():
        assert mine.execute(GetRoom, {"id": "room-1"}).room.id == "room-1"
    assert len(mine.rath.sent) == 1


async def test_asubscribe_goes_through_the_client() -> None:
    mine = client()
    events = [e async for e in mine.asubscribe(GetRoom, {"id": "room-1"})]
    assert [e.room.id for e in events] == ["room-1"]
    assert get_origin(events[0].room).client is mine


async def test_unset_arguments_stay_off_the_wire() -> None:
    """alpaka's wire behaviour, kept through the rewrite: exclude_unset."""
    mine = client()
    await mine.aexecute(GetRoom, {"id": "room-1"})
    assert mine.rath.sent == [{"id": "room-1"}]


def test_every_operation_is_a_method_and_the_client_has_only_its_fields() -> None:
    assert set(Alpaka.model_fields) == {"rath", "task_token", "llm_url", "tokens"}
    for name in ("aget_room", "get_room", "achat", "astart_message", "awatch_room"):
        assert callable(getattr(Alpaka, name))


async def test_a_generated_method_goes_through_its_own_client() -> None:
    """Two clients side by side: each call lands on the one it was made on."""
    calls: list[str] = []

    def rath(name: str) -> AlpakaRath:
        async def room(operation: Operation) -> dict[str, Any]:
            calls.append(name)
            return {"id": operation.variables["id"], "title": "t", "description": None}

        return AlpakaRath(link=AsyncMockLink(query_resolver={"room": room}))

    a, b = client(rath("a")), client(rath("b"))
    async with a.rath, b.rath:
        fetched = await b.aget_room(id="5")
        await a.aget_room(id="6")

    assert calls == ["b", "a"]
    assert fetched.id == "5"


async def test_stream_into_room_writes_through_the_client_it_is_given() -> None:
    """Start, every append and the finish go through the one client passed in."""
    calls: list[str] = []

    def recording(name: str) -> AlpakaRath:
        def answer(kind: str) -> Any:
            async def resolve(operation: Operation) -> dict[str, Any]:
                calls.append(f"{name}:{kind}")
                return {
                    "id": "1",
                    "text": "",
                    "isStreaming": kind != "finishMessage",
                    "agent": {"id": "7"},
                    "attachedStructures": [],
                }

            return resolve

        kinds = ("startMessage", "appendMessage", "finishMessage")
        return AlpakaRath(link=AsyncMockLink(mutation_resolver={k: answer(k) for k in kinds}))

    a, b = client(recording("a")), client(recording("b"))
    async with a.rath, b.rath:
        async with stream_into_room(a, room="3", agent_id="assistant", flush_chars=1) as reply:
            assert reply.client is a
            await reply.append("hello")

    assert calls == ["a:startMessage", "a:appendMessage", "a:finishMessage"]


def test_stream_into_room_needs_a_client() -> None:
    """There is no ambient fallback: the client is a required positional argument."""
    with pytest.raises(TypeError):
        stream_into_room(room="3", agent_id="assistant")  # type: ignore[call-arg]



@pytest.mark.asyncio
async def test_a_task_view_stamps_its_token() -> None:
    from types import SimpleNamespace

    base = Alpaka.model_construct(rath=FakeRath(), task_token=None)
    view = base.for_task(SimpleNamespace(token="tok"))

    await base.aexecute(GetRoom, {"id": "r"})
    await view.aexecute(GetRoom, {"id": "r"})

    assert base.rath.headers == [None, {TASK_HEADER: "tok"}]
    assert base.task_token is None and view.rath is base.rath


@pytest.mark.asyncio
async def test_the_ambient_task_is_stamped_without_a_view() -> None:
    """One shared client attributes each call to whatever task is running."""
    from types import SimpleNamespace

    from rath.task import task_scope

    base = Alpaka.model_construct(rath=FakeRath(), task_token=None)

    await base.aexecute(GetRoom, {"id": "r"})
    with task_scope(SimpleNamespace(token="tok")):
        await base.aexecute(GetRoom, {"id": "r"})

    assert base.rath.headers == [None, {TASK_HEADER: "tok"}]
    assert base.task_token is None, "the shared client is never changed"


@pytest.mark.asyncio
async def test_a_task_named_at_the_call_beats_the_ambient_one() -> None:
    from types import SimpleNamespace

    from rath.task import task_scope

    base = Alpaka.model_construct(rath=FakeRath(), task_token=None)
    with task_scope(SimpleNamespace(token="ambient")):
        await base.aexecute(GetRoom, {"id": "r"}, task=SimpleNamespace(token="explicit"))

    assert base.rath.headers == [{TASK_HEADER: "explicit"}]
