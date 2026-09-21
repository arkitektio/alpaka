# alpaka

[![codecov](https://codecov.io/gh/jhnnsrs/akuire/branch/master/graph/badge.svg?token=UGXEA2THBV)](https://codecov.io/gh/jhnnsrs/akuire)
[![PyPI version](https://badge.fury.io/py/akuire.svg)](https://pypi.org/project/akuire/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://pypi.org/project/akuire/)
![Maintainer](https://img.shields.io/badge/maintainer-jhnnsrs-blue)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/akuire.svg)](https://pypi.python.org/pypi/akuire/)
[![PyPI status](https://img.shields.io/pypi/status/akuire.svg)](https://pypi.python.org/pypi/akuire/)

Alpaka is the arkitekt LLM gateway client. The alpaka server fronts litellm —
Ollama, OpenAI, Anthropic, OpenRouter and friends — behind arkitekt auth, a
per-organization model registry, and an **OpenAI-compatible REST API**. This
package deliberately does not reinvent a chat SDK: it brokers the connection
and hands you the client you already know.

## Chat: use the SDK you like, tunneled through alpaka

```python
import alpaka
from arkitekt import App, connect

with connect(App("chat", services=[alpaka.alpaka_service])) as rt:
    client = rt.require(alpaka.Alpaka).openai  # a real openai.OpenAI, pointed at your alpaka server

    response = client.chat.completions.create(
        model="alpaka/default",  # or "openrouter/gpt-4", "ollama/llama3", a registry id...
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
```

Requires the `openai` extra: `pip install alpaka[openai]`. The client builds the
SDK client once, on first use, from the endpoint and token loader its service
gave it; `alpaka_client.aopenai` is the `openai.AsyncOpenAI` twin. Auth tokens
are re-read on every request, so long-running sessions survive token expiry.
For SDK options of your own, `alpaka.build_openai(alpaka_client.llm_url,
alpaka_client.tokens, max_retries=7)` builds a fresh one the same way.

Anything else that speaks the OpenAI wire format — LangChain, curl, a JS app —
can tunnel too:

```python
endpoint = rt.require(alpaka.Alpaka).get_endpoint()
print(endpoint.base_url)  # .../llm/v1
print(endpoint.api_key)   # your current arkitekt token (expires!)
```

Model names accept `provider/model-id`, a bare registry `model-id`, a database
id, or the `alpaka/default` / `alpaka/default-chat` / `alpaka/default-embedding`
defaults configured on the server.

## Streaming a reply into a room

Agents live on clients: the server never calls an LLM on behalf of a room. When
you have a stream of tokens, forward it into the room and every subscriber sees
it grow.

```python
import alpaka
from arkitekt import App, connect

async with connect(App("chat", services=[alpaka.alpaka_service])) as rt:
    alpaka_client = rt.require(alpaka.Alpaka)
    client = alpaka_client.aopenai
    completion = await client.chat.completions.create(
        model="alpaka/default",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    # The alpaka client to write through is passed in: nothing is looked up.
    async with alpaka.stream_into_room(alpaka_client, room=room.id, agent_id="assistant") as reply:
        async for chunk in completion:
            await reply.append(chunk.choices[0].delta.content or "")
```

`stream_into_room` opens the message, coalesces deltas (~30 characters or
~150 ms, so you do not pay a mutation per token), sends them in order, and always
finishes the message on the way out — with the full accumulated text, so a delta
lost on the way is repaired, and a raising body cannot leave a message
`isStreaming` forever. Watch a room with `service.watch_room` / `service.awatch_room`: every
event carries a `kind` (`MESSAGE_CREATED`, `MESSAGE_UPDATED`, `MESSAGE_FINISHED`,
`JOIN`, `LEAVE`). A subscription only sees what happens after it joins.

The underlying mutations (`service.start_message` / `append_message` /
`finish_message`, like every generated operation a method of the `Alpaka`
client) are there if you want to drive it yourself; the server also has a
one-frame-per-token websocket at `/kammer/stream/` for clients that need it.

## Testing

```bash
uv run pytest -m "not integration"   # fast unit layer: no docker, <1s
uv run pytest                        # full suite: spins the dokker compose stack
```

The unit layer covers the tunnel broker (endpoint resolution and per-request
token refresh, driven by a hot-plugged `fakts.testing.TestingFakts` —
a real Fakts entered on the normal context, no monkeypatching — with HTTP
through httpx.MockTransport) and the GraphQL wire
contract (camelCase aliases, unset-field omission, @oneOf order variants)
through a rath `AsyncMockLink`, and the room-streaming helper (coalescing,
ordering, and the always-finish guarantee). The `integration`-marked tests run
the same operations — plus the OpenAI-compatible `/llm/v1` endpoint with the real
openai SDK, and the room subscription end to end — against the composed
`jhnnsrs/alpaka:next` image.

That image is whatever was last published, so server changes are invisible here
until it is rebuilt. To test against a local `alpaka-server` checkout instead,
copy `tests/integration/docker-compose.local.example.yml` to
`docker-compose.local.yml` (gitignored) and point it at your checkout — the conftest picks
it up when it exists, and CI keeps testing the published image.

## GraphQL client

The generated GraphQL client (every operation in `alpaka.api.schema` is a
method of `AlpakaApi`, which the `Alpaka` client mixes in: `service.aget_room(id)`)
covers the registry and
collaboration surface: listing and searching `LLMModel`s and providers, rooms
and messages (including the streaming mutations above), and the ChromaDB
vector-collection RAG API. The `chat` mutation
exists for rekuest-registered functions that receive an `LLMModel` picker, but
for interactive chat and streaming prefer the tunnel above.
