"""Tunnel native LLM SDKs through the alpaka service.

Alpaka's chat surface is deliberately not its own SDK: the server exposes an
OpenAI-compatible REST API (``/llm/v1/...``) behind arkitekt auth, so people
use the client they already know and alpaka only brokers the connection —
resolving the endpoint from fakts and injecting the current auth token.

Three entry points:

- :func:`get_endpoint` — the base URL and a current token, for any
  OpenAI-compatible consumer in any language (curl, LangChain, a JS app).
- :func:`openai` / :func:`aopenai` — a configured ``openai.OpenAI`` /
  ``openai.AsyncOpenAI`` (requires the ``alpaka[openai]`` extra). Tokens are
  re-read from fakts on every request, so long-running sessions survive token
  expiry.

Model names accept the server's registry forms: ``provider/model-id`` (e.g.
``openrouter/gpt-4``), a bare ``model-id``, or ``alpaka/default`` /
``alpaka/default-chat`` / ``alpaka/default-embedding`` for the configured
default model.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

try:
    import httpx
except ImportError:  # openai >= 3 environments may ship only httpx2
    import httpx2 as httpx  # type: ignore[no-redef]

from fakts import get_current_fakts

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

__all__ = [
    "AlpakaEndpoint",
    "get_endpoint",
    "aget_endpoint",
    "openai",
    "aopenai",
    "alpakaAI",
]


@dataclass(frozen=True)
class AlpakaEndpoint:
    """An OpenAI-compatible endpoint on the alpaka server.

    ``api_key`` is the fakts token as of the moment this was created; it
    expires. For anything long-running prefer :func:`openai` / :func:`aopenai`,
    which refresh the token per request.
    """

    base_url: str
    api_key: str


def get_endpoint() -> AlpakaEndpoint:
    """Resolve the alpaka OpenAI-compatible endpoint from the current fakts
    context (requires an active arkitekt/fakts session, e.g. ``with easy():``)."""
    fakts = get_current_fakts()
    return AlpakaEndpoint(
        base_url=fakts.get_alias("alpaka").to_http_path("/llm/v1"),
        api_key=fakts.get_token(),
    )


async def aget_endpoint() -> AlpakaEndpoint:
    """Async twin of :func:`get_endpoint`, for callers inside an event loop."""
    fakts = get_current_fakts()
    alias = await fakts.aget_alias("alpaka")
    return AlpakaEndpoint(
        base_url=alias.to_http_path("/llm/v1"),
        api_key=await fakts.aget_token(),
    )


class _FaktsBearerAuth(httpx.Auth):
    """Per-request bearer auth that re-reads the fakts token, so an SDK client
    built once keeps working after the token it was created with expires."""

    def sync_auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {get_current_fakts().get_token()}"
        yield request

    async def async_auth_flow(self, request: httpx.Request):
        token = await get_current_fakts().aget_token()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def openai(**kwargs: Any) -> "OpenAI":
    """A native ``openai.OpenAI`` client tunneled through alpaka.

    Extra ``kwargs`` are forwarded to the SDK constructor. Requires the
    ``alpaka[openai]`` extra.
    """
    from openai import OpenAI

    fakts = get_current_fakts()
    kwargs.setdefault(
        "http_client", httpx.Client(auth=_FaktsBearerAuth(), timeout=httpx.Timeout(600.0))
    )
    return OpenAI(
        base_url=fakts.get_alias("alpaka").to_http_path("/llm/v1"),
        # The real credential is injected per request by _FaktsBearerAuth;
        # the SDK just refuses to construct without an api_key.
        api_key="managed-by-alpaka",
        **kwargs,
    )


async def aopenai(**kwargs: Any) -> "AsyncOpenAI":
    """A native ``openai.AsyncOpenAI`` client tunneled through alpaka.

    Async (``await alpaka.aopenai()``) because resolving the alias from
    inside a running event loop must not cross the sync koil bridge — the
    sync :func:`openai` is for synchronous call sites. Extra ``kwargs`` are
    forwarded to the SDK constructor. Requires the ``alpaka[openai]`` extra.
    """
    from openai import AsyncOpenAI

    fakts = get_current_fakts()
    alias = await fakts.aget_alias("alpaka")
    kwargs.setdefault(
        "http_client", httpx.AsyncClient(auth=_FaktsBearerAuth(), timeout=httpx.Timeout(600.0))
    )
    return AsyncOpenAI(
        base_url=alias.to_http_path("/llm/v1"),
        api_key="managed-by-alpaka",
        **kwargs,
    )


def alpakaAI() -> "OpenAI":
    """Deprecated alias for :func:`openai`."""
    warnings.warn(
        "alpakaAI() is deprecated; use alpaka.openai() (or alpaka.aopenai()) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return openai()
