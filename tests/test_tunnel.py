"""Unit tests for the SDK tunnel: the client owns its endpoint and its ``openai``.

No docker, no monkeypatching: a real ``fakts.testing.TestingFakts`` stands in
as the client's token loader, as the runtime's fakts does, and HTTP goes
through ``httpx.MockTransport``. Because TestingFakts runs the real token
machinery (locks, expiry, refresh), the rotation tests prove the true
expiry->refetch path, not a fake counter.

Needs the ``openai`` extra (openai + fakts); skipped where absent.
"""

import httpx
import pytest

pytest.importorskip("openai")
fakts = pytest.importorskip("fakts")
if not hasattr(fakts, "build_testing_fakts"):  # pragma: no cover
    pytest.skip("installed fakts predates TestingFakts", allow_module_level=True)

from fakts.testing import build_testing_fakts  # noqa: E402

from alpaka.alpaka import Alpaka  # noqa: E402
from alpaka.tunnel import PLACEHOLDER_API_KEY, build_async_openai, build_openai  # noqa: E402

LLM_URL = "http://testserver/llm/v1"

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


def client_with(tokens: object) -> Alpaka:
    """An Alpaka as the service builds it, minus the rath: endpoint and tokens."""
    return Alpaka.model_construct(rath=None, task_token=None, llm_url=LLM_URL, tokens=tokens)


@pytest.fixture
def tokens():
    """A hot-plugged fakts with a static, never-expiring token, as the token loader."""
    with build_testing_fakts(aliases={"alpaka": "http://testserver"}, token="static-token") as f:
        yield f


def test_get_endpoint_is_the_clients_url_and_current_token(tokens):
    endpoint = client_with(tokens).get_endpoint()
    assert endpoint.base_url == LLM_URL
    assert endpoint.api_key == "static-token"


async def test_aget_endpoint_inside_a_running_loop():
    async with build_testing_fakts(
        aliases={"alpaka": "http://testserver"}, token="static-token"
    ) as tokens:
        endpoint = await client_with(tokens).aget_endpoint()
    assert endpoint.base_url == LLM_URL
    assert endpoint.api_key == "static-token"


def test_openai_is_built_once_lazily_and_points_at_the_tunnel(tokens):
    """No credential is handed to the SDK -- auth is per request -- and nothing
    is fetched until a request is made."""
    client = client_with(tokens)
    sdk = client.openai
    assert str(sdk.base_url).rstrip("/") == LLM_URL
    assert sdk.api_key == PLACEHOLDER_API_KEY
    assert tokens.token_fetches == 0
    assert client.openai is sdk
    assert client.aopenai is client.aopenai


def test_sync_requests_refresh_token_each_time():
    """Every request re-reads the token through the real expiry machinery, so a
    client built before a token expired keeps working."""
    seen_auth: list[str] = []
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=CANNED_COMPLETION)

    with build_testing_fakts(
        aliases={"alpaka": "http://testserver"}, tokens=TOKEN_SEQUENCE, token_lifetime=0
    ) as tokens:
        sdk = build_openai(LLM_URL, tokens, transport=httpx.MockTransport(handler))
        messages = [{"role": "user", "content": "hi"}]
        first = sdk.chat.completions.create(model="alpaka/default", messages=messages)
        second = sdk.chat.completions.create(model="alpaka/default", messages=messages)

    assert first.choices[0].message.content == "hello"
    assert second.choices[0].message.content == "hello"
    assert seen_auth == ["Bearer tok-1", "Bearer tok-2"]
    assert seen_urls == [f"{LLM_URL}/chat/completions"] * 2


async def test_async_requests_refresh_token_each_time():
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers["Authorization"])
        return httpx.Response(200, json=CANNED_COMPLETION)

    async with build_testing_fakts(
        aliases={"alpaka": "http://testserver"}, tokens=TOKEN_SEQUENCE, token_lifetime=0
    ) as tokens:
        sdk = build_async_openai(LLM_URL, tokens, transport=httpx.MockTransport(handler))
        messages = [{"role": "user", "content": "hi"}]
        await sdk.chat.completions.create(model="alpaka/default", messages=messages)
        await sdk.chat.completions.create(model="alpaka/default", messages=messages)

    assert seen_auth == ["Bearer tok-1", "Bearer tok-2"]


def test_constructor_kwargs_forwarded(tokens):
    assert build_openai(LLM_URL, tokens, max_retries=7).max_retries == 7


def test_a_foreign_http_client_is_refused(tokens):
    """Passing one would silently drop the bearer auth; the transport is the seam."""
    with pytest.raises(TypeError, match="transport="):
        build_openai(LLM_URL, tokens, http_client=httpx.Client())


def test_a_task_view_shares_the_tunnel(tokens):
    """Inside an action the injected client is a per-task view: same endpoint, same SDK."""
    client = client_with(tokens)
    sdk = client.openai
    view = client.for_task(type("T", (), {"token": "t"})())
    assert view.openai is sdk
    assert view.get_endpoint().api_key == "static-token"
