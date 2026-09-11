"""Unit tests for the SDK tunnel (``alpaka.tunnel``).

No docker, no monkeypatching: a real ``fakts.testing.TestingFakts`` is
entered as a context manager, which publishes it on the genuine
``current_fakts`` contextvar — exactly how production code finds fakts —
and HTTP goes through ``httpx.MockTransport``. Because TestingFakts runs the
real token machinery (locks, expiry, refresh), the rotation tests prove the
true expiry→refetch path, not a fake counter.

Token accounting note: resolving an alias spends one token fetch (fakts takes
the report token up front — real behavior), so the first token a request sees
is the *second* one in the configured sequence.

Needs the ``openai`` extra (openai + fakts-next); skipped where absent.
"""

import httpx
import pytest

pytest.importorskip("openai")
fakts = pytest.importorskip("fakts")
if not hasattr(fakts, "build_testing_fakts"):  # pragma: no cover
    pytest.skip("installed fakts-next predates TestingFakts", allow_module_level=True)

from fakts.testing import build_testing_fakts  # noqa: E402

import alpaka.tunnel as tunnel  # noqa: E402


CANNED_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "alpaka/default",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

TOKEN_SEQUENCE = [f"tok-{i}" for i in range(1, 6)]


@pytest.fixture
def fakts():
    """A hot-plugged fakts with a static, never-expiring token."""
    with build_testing_fakts(
        aliases={"alpaka": "http://testserver"}, token="static-token"
    ) as f:
        yield f


@pytest.fixture
def rotating_fakts():
    """A hot-plugged fakts whose token expires immediately, so every
    ``get_token`` call renews — the long-running-session shape."""
    with build_testing_fakts(
        aliases={"alpaka": "http://testserver"},
        tokens=TOKEN_SEQUENCE,
        token_lifetime=0,
    ) as f:
        yield f


def test_get_endpoint_resolves_base_url_and_current_token(fakts):
    endpoint = tunnel.get_endpoint()
    assert endpoint.base_url == "http://testserver/llm/v1"
    assert endpoint.api_key == "static-token"


async def test_aget_endpoint_resolves_inside_a_running_loop():
    async with build_testing_fakts(
        aliases={"alpaka": "http://testserver"}, token="static-token"
    ):
        endpoint = await tunnel.aget_endpoint()
    assert endpoint.base_url == "http://testserver/llm/v1"
    assert endpoint.api_key == "static-token"


def test_openai_client_points_at_tunnel(fakts):
    """The factory resolves the alias (which spends fakts' report-token
    fetch) but hands the SDK no real credential — auth is per request."""
    client = tunnel.openai()
    assert str(client.base_url).rstrip("/") == "http://testserver/llm/v1"
    assert client.api_key == "managed-by-alpaka"
    assert fakts.token_fetches == 1  # alias resolution only


def test_sync_requests_refresh_token_each_time(rotating_fakts):
    """Every request re-reads the fakts token through the real expiry
    machinery, so a client built before a token expired keeps working."""
    seen_auth: list[str] = []
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=CANNED_COMPLETION)

    client = tunnel.openai(
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler), auth=tunnel._FaktsBearerAuth()
        )
    )
    messages = [{"role": "user", "content": "hi"}]
    first = client.chat.completions.create(model="alpaka/default", messages=messages)
    second = client.chat.completions.create(model="alpaka/default", messages=messages)

    assert first.choices[0].message.content == "hello"
    assert second.choices[0].message.content == "hello"
    # tok-1 went to alias resolution at construction; each request renews.
    assert seen_auth == ["Bearer tok-2", "Bearer tok-3"]
    assert seen_urls == ["http://testserver/llm/v1/chat/completions"] * 2


async def test_async_requests_refresh_token_each_time():
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        return httpx.Response(200, json=CANNED_COMPLETION)

    async with build_testing_fakts(
        aliases={"alpaka": "http://testserver"},
        tokens=TOKEN_SEQUENCE,
        token_lifetime=0,
    ):
        client = await tunnel.aopenai(
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler), auth=tunnel._FaktsBearerAuth()
            )
        )
        messages = [{"role": "user", "content": "hi"}]
        await client.chat.completions.create(model="alpaka/default", messages=messages)
        await client.chat.completions.create(model="alpaka/default", messages=messages)

    assert seen_auth == ["Bearer tok-2", "Bearer tok-3"]


def test_constructor_kwargs_forwarded(fakts):
    client = tunnel.openai(max_retries=7)
    assert client.max_retries == 7


def test_alpaka_ai_is_deprecated_alias(fakts):
    with pytest.deprecated_call():
        client = tunnel.alpakaAI()
    assert str(client.base_url).rstrip("/") == "http://testserver/llm/v1"
