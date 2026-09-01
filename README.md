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
from arkitekt_next import easy

with easy():
    client = alpaka.openai()  # a real openai.OpenAI, pointed at your alpaka server

    response = client.chat.completions.create(
        model="alpaka/default",  # or "openrouter/gpt-4", "ollama/llama3", a registry id...
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="")
```

Requires the `openai` extra: `pip install alpaka[openai]`. There is an async
twin — `client = await alpaka.aopenai()` (async because alias resolution
inside a running event loop must not cross the sync koil bridge), plus
`await alpaka.aget_endpoint()`. Auth tokens are re-read from fakts on every
request, so long-running sessions survive token expiry.

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

## Testing

```bash
uv run pytest -m "not integration"   # fast unit layer: no docker, <1s
uv run pytest                        # full suite: spins the dokker compose stack
```

The unit layer covers the tunnel broker (endpoint resolution and per-request
token refresh, driven by a hot-plugged `fakts_next.testing.TestingFakts` —
a real Fakts entered on the normal context, no monkeypatching — with HTTP
through httpx.MockTransport) and the GraphQL wire
contract (camelCase aliases, unset-field omission, @oneOf order variants)
through a rath `AsyncMockLink`. The `integration`-marked tests run the same
operations — plus the OpenAI-compatible `/llm/v1` endpoint with the real
openai SDK — against the composed `jhnnsrs/alpaka:next` image.

## GraphQL client

The generated GraphQL client (`alpaka.api.schema`) covers the registry and
collaboration surface: listing and searching `LLMModel`s and providers, rooms
and messages, and the ChromaDB vector-collection RAG API. The `chat` mutation
exists for rekuest-registered functions that receive an `LLMModel` picker, but
for interactive chat and streaming prefer the tunnel above.
