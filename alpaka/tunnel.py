"""Tunnel native LLM SDKs through the alpaka service.

Alpaka's chat surface is deliberately not its own SDK: the server exposes an
OpenAI-compatible REST API (``/llm/v1/...``) behind arkitekt auth, so people
use the client they already know and alpaka only brokers the connection --
the endpoint the service builder resolved, and the current auth token.

Three entry points:

- ``Alpaka.openai`` / ``Alpaka.aopenai`` -- the client's own ``openai.OpenAI`` /
  ``openai.AsyncOpenAI``, built once, lazily, from the endpoint and token loader
  the client was built with (requires the ``alpaka[openai]`` extra). Tokens are
  re-read on every request, so long-running sessions survive token expiry.
- ``Alpaka.get_endpoint()`` / ``aget_endpoint()`` -- the base URL and a current
  token, for any OpenAI-compatible consumer in any language (curl, LangChain, a
  JS app).
- :func:`build_openai` / :func:`build_async_openai` -- the same construction
  with your own SDK options, for when the client's cached one is not what you
  want.

Model names accept the server's registry forms: ``provider/model-id`` (e.g.
``openrouter/gpt-4``), a bare ``model-id``, or ``alpaka/default`` /
``alpaka/default-chat`` / ``alpaka/default-embedding`` for the configured
default model.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from koil import unkoil

try:
    import httpx
except ImportError:  # openai >= 3 environments may ship only httpx2
    import httpx2 as httpx  # type: ignore[no-redef]


if TYPE_CHECKING:
    from fakts import TokenLoader
    from openai import AsyncOpenAI, OpenAI

__all__ = [
    "AlpakaEndpoint",
    "TokenAuth",
    "build_openai",
    "build_async_openai",
]


@dataclass(frozen=True)
class AlpakaEndpoint:
    """An OpenAI-compatible endpoint on the alpaka server.

    ``api_key`` is the token as of the moment this was created; it expires. For
    anything long-running prefer the client's ``openai`` / ``aopenai``, which
    refresh the token per request.
    """

    base_url: str
    api_key: str


class TokenAuth(httpx.Auth):
    """Per-request bearer auth that re-reads the token, so an SDK client built
    once keeps working after the token it was created with expires."""

    def __init__(self, tokens: "TokenLoader") -> None:
        self.tokens = tokens

    def sync_auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {unkoil(self.tokens.aget_token)}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        token = await self.tokens.aget_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


#: The SDK refuses to construct without an api_key; the real credential is
#: injected per request by :class:`TokenAuth`.
PLACEHOLDER_API_KEY = "managed-by-alpaka"


def _refuse_http_client(kwargs: dict[str, Any]) -> None:
    if "http_client" in kwargs:
        raise TypeError(
            "build_openai builds the http client itself, so the bearer auth is "
            "never lost. Pass `transport=` to route its requests, or build the "
            "SDK client yourself with TokenAuth(tokens)."
        )


def build_openai(
    base_url: str,
    tokens: "TokenLoader",
    *,
    transport: "httpx.BaseTransport | None" = None,
    **kwargs: Any,
) -> "OpenAI":
    """A native ``openai.OpenAI`` client tunneled through alpaka.

    Args:
        base_url: The OpenAI-compatible endpoint, ``<alpaka>/llm/v1``.
        tokens: Where the bearer token comes from, re-read on every request.
        transport: An ``httpx`` transport to route requests through (a
            ``MockTransport`` in tests). The http client itself is always built
            here, with the bearer auth.
        **kwargs: Forwarded to the SDK constructor (``max_retries=``, ...).

    Returns:
        The client. Requires the ``alpaka[openai]`` extra.

    Raises:
        TypeError: If ``http_client`` is passed; it would drop the bearer auth.
    """
    from openai import OpenAI

    _refuse_http_client(kwargs)
    http_client = httpx.Client(
        auth=TokenAuth(tokens), timeout=httpx.Timeout(600.0), transport=transport
    )
    return OpenAI(
        base_url=base_url, api_key=PLACEHOLDER_API_KEY, http_client=http_client, **kwargs
    )


def build_async_openai(
    base_url: str,
    tokens: "TokenLoader",
    *,
    transport: "httpx.AsyncBaseTransport | None" = None,
    **kwargs: Any,
) -> "AsyncOpenAI":
    """A native ``openai.AsyncOpenAI`` client tunneled through alpaka.

    See :func:`build_openai`; the transport is an async one.
    """
    from openai import AsyncOpenAI

    _refuse_http_client(kwargs)
    http_client = httpx.AsyncClient(
        auth=TokenAuth(tokens), timeout=httpx.Timeout(600.0), transport=transport
    )
    return AsyncOpenAI(
        base_url=base_url, api_key=PLACEHOLDER_API_KEY, http_client=http_client, **kwargs
    )
