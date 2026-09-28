# alpaka

[![PyPI version](https://badge.fury.io/py/alpaka.svg)](https://pypi.org/project/alpaka/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/alpaka.svg)](https://pypi.python.org/pypi/alpaka/)
![Maintainer](https://img.shields.io/badge/maintainer-jhnnsrs-blue)

Alpaka is the [Arkitekt](https://arkitekt.live) LLM gateway client. The alpaka server fronts litellm —
Ollama, OpenAI, Anthropic, OpenRouter and friends — behind arkitekt auth, a
per-organization model registry, and an **OpenAI-compatible REST API**. This
package deliberately does not reinvent a chat SDK: it brokers the connection
and hands you the client you already know.

## Installation

```sh
pip install "alpaka[openai]"
```

The `openai` extra brings the tunnel (the `openai` SDK, `httpx`, `fakts`).

## Usage

Every alpaka operation is a method of the `Alpaka` client, in a blocking and an
`a`-prefixed async flavour. Its `openai` / `aopenai` properties are real
`openai.OpenAI` / `openai.AsyncOpenAI` clients pointed at your alpaka server.

### In an arkitekt app

The client is injected by annotation. Add the service to your app and ask for
`alpaka: Alpaka`:

```python
from arkitekt import App, run
from alpaka import Alpaka, alpaka_service

app = App("summarize", "0.1.0", services=[alpaka_service])


@app.action
def summarize(text: str, alpaka: Alpaka) -> str:
    """Summarize"""
    response = alpaka.openai.chat.completions.create(
        model="alpaka/default",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    run(app)
```

Rooms, models and messages travel between actions by id (`@alpaka/room`,
`@alpaka/llmmodel`, `@alpaka/message`), so an action can take and return them
directly.

### From a script

```python
from arkitekt import easy
from alpaka import alpaka_service

with easy("chat", alpaka_service) as alpaka:
    response = alpaka.openai.chat.completions.create(
        model="alpaka/default",  # or "openrouter/gpt-4", "ollama/llama3", a registry id...
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
```

The client builds the SDK client once, on first use, from the endpoint and
token loader its service gave it. Auth tokens are re-read on every request, so
long-running sessions survive token expiry. For SDK options of your own,
`build_openai(alpaka.llm_url, alpaka.tokens, max_retries=7)` (and
`build_async_openai`, both importable from `alpaka`) builds a fresh one the same way.

Anything else that speaks the OpenAI wire format — LangChain, curl, a JS app —
can tunnel too:

```python
endpoint = alpaka.get_endpoint()
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
from arkitekt import aeasy
from alpaka import alpaka_service, stream_into_room

async with aeasy("chat", alpaka_service) as alpaka:
    room = await alpaka.acreate_room(title="Demo")
    completion = await alpaka.aopenai.chat.completions.create(
        model="alpaka/default",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    # The alpaka client to write through is passed in: nothing is looked up.
    async with stream_into_room(alpaka, room=room.id, agent_id="assistant") as reply:
        async for chunk in completion:
            await reply.append(chunk.choices[0].delta.content or "")
```

`stream_into_room` opens the message, coalesces deltas (~30 characters or
~150 ms, so you do not pay a mutation per token), sends them in order, and always
finishes the message on the way out — with the full accumulated text, so a delta
lost on the way is repaired, and a raising body cannot leave a message
`isStreaming` forever. Watch a room with `alpaka.watch_room` / `alpaka.awatch_room`: every
event carries a `kind` (`MESSAGE_CREATED`, `MESSAGE_UPDATED`, `MESSAGE_FINISHED`,
`JOIN`, `LEAVE`). A subscription only sees what happens after it joins.

The underlying mutations (`alpaka.start_message` / `append_message` /
`finish_message`) are there if you want to drive it yourself; the server also has a
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
method of `AlpakaApi`, which the `Alpaka` client mixes in: `alpaka.aget_room(id)`)
covers the registry and
collaboration surface: listing and searching `LLMModel`s and providers, rooms
and messages (including the streaming mutations above), and the ChromaDB
vector-collection RAG API. The `chat` mutation
exists for arkitekt actions that receive an `LLMModel` picker, but
for interactive chat and streaming prefer the tunnel above.
