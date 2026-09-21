"""Wire-serialization unit tests for the generated GraphQL client.

No docker, no network: an ``AsyncMockLink`` terminates the rath chain, so these
prove exactly what goes onto the wire (camelCase aliases, unset-field omission,
``@oneOf`` order variants serializing to a single key) and that responses parse
back into the typed models — the regressions a turms regeneration is most
likely to introduce. The dokker integration suite then covers the same
operations against a real server.
"""

from typing import Any

from rath.links.testing.mock import AsyncMockLink
from rath.operation import Operation

from alpaka.api.schema import (
    ChatMessageInput,
    Ordering,
    Role,
    RoomOrderCreatedAt,
    ThinkingBlockType,
)
from alpaka.alpaka import Alpaka
from alpaka.rath import AlpakaRath

from .conftest import StaticTokens


def client(rath: AlpakaRath) -> Alpaka:
    """An Alpaka over the given rath, with a static tunnel endpoint."""
    return Alpaka(rath=rath, llm_url="http://testserver/llm/v1", tokens=StaticTokens())



def chat_payload(**choice_overrides: Any) -> dict:
    """A minimal valid payload for the ``ChatResponse`` fragment."""
    choice = {
        "index": 0,
        "finishReason": "stop",
        "message": {"role": "ASSISTANT", "content": "Hello there"},
    }
    choice.update(choice_overrides)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openrouter/gpt-4",
        "usage": {"promptTokens": 10, "completionTokens": 20, "totalTokens": 30},
        "choices": [choice],
    }


async def test_chat_serializes_camel_case_and_omits_unset() -> None:
    """Snake_case kwargs land on the wire under their camelCase aliases, and
    parameters left unset are omitted entirely rather than sent as null."""
    captured: dict = {}

    async def resolve_chat(operation: Operation) -> dict:
        captured.update(operation.variables)
        return chat_payload()

    async with AlpakaRath(link=AsyncMockLink(mutation_resolver={"chat": resolve_chat})) as rath:
        await client(rath).achat(
            messages=[ChatMessageInput(role=Role.USER, content="hi")],
            max_tokens=5,
            top_p=0.9,
            response_format={"type": "json_object"},
        )

    sent = captured["input"]
    assert sent["maxTokens"] == 5
    assert sent["topP"] == 0.9
    assert sent["responseFormat"] == {"type": "json_object"}
    # The message dumps only what was set, with the enum's wire value. (The
    # tuple-typed field dumps as a tuple; JSON encoding turns it into a list.)
    assert list(sent["messages"]) == [{"role": "USER", "content": "hi"}]
    # Unset parameters must not appear at all.
    for absent in ("temperature", "tools", "toolChoice", "stop", "n", "model"):
        assert absent not in sent


async def test_chat_response_parses_tools_thinking_and_usage() -> None:
    """An OpenAI-shaped response with tool calls and thinking blocks parses
    into the typed models, and the ``to_string`` trait reads the first choice."""

    async def resolve_chat(operation: Operation) -> dict:
        return chat_payload(
            reasoningContent="thinking out loud",
            thinkingBlocks=[
                {"type": "THINKING", "thinking": "step 1", "signature": None}
            ],
            message={
                "role": "ASSISTANT",
                "content": "Hello there",
                "toolCalls": [
                    {
                        "id": "call_1",
                        "type": "FUNCTION",
                        "function": {"name": "get_weather", "arguments": '{"city": "x"}'},
                    }
                ],
            },
        )

    async with AlpakaRath(link=AsyncMockLink(mutation_resolver={"chat": resolve_chat})) as rath:
        response = await client(rath).achat(
            messages=[ChatMessageInput(role=Role.USER, content="hi")]
        )

    assert response.to_string() == "Hello there"
    assert response.usage is not None and response.usage.total_tokens == 30
    choice = response.choices[0]
    assert choice.reasoning_content == "thinking out loud"
    assert choice.thinking_blocks is not None
    assert choice.thinking_blocks[0].type == ThinkingBlockType.THINKING
    assert choice.thinking_blocks[0].thinking == "step 1"
    tool_call = choice.message.tool_calls[0]
    assert tool_call.id == "call_1"
    assert tool_call.function.name == "get_weather"


async def test_oneof_order_variant_serializes_to_single_key() -> None:
    """A ``@oneOf`` order variant (``RoomOrderCreatedAt``) must serialize to
    exactly one aliased key — the shape the server's oneOf validation demands."""
    captured: dict = {}

    async def resolve_rooms(operation: Operation) -> list:
        captured.update(operation.variables)
        return []

    async with AlpakaRath(link=AsyncMockLink(query_resolver={"rooms": resolve_rooms})) as rath:
        await client(rath).alist_rooms(order=[RoomOrderCreatedAt(created_at=Ordering.ASC)])

    assert captured["order"] == [{"createdAt": "ASC"}]
