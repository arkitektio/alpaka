"""The base client for alpaka"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

from typing import TYPE_CHECKING

from koil import unkoil, unkoil_gen
from koil.composition import Composition
from pydantic import Field, PrivateAttr
from rath.origin import origin_context
from rath.turms.funcs import TOperation

from alpaka.api.schema import AlpakaApi
from alpaka.rath import AlpakaRath

if TYPE_CHECKING:
    # The SDK is the optional `alpaka[openai]` extra: imported for the type
    # checker only, so the client's fields stay typed without it installed.
    from openai import AsyncOpenAI, OpenAI

    from alpaka.tunnel import AlpakaEndpoint

TASK_HEADER = "Rekuest-Task"
"""The header a per-task client view stamps its provenance token under."""


class Alpaka(Composition, AlpakaApi):
    """Alpaka

    Every alpaka operation is a method of it (``alpaka.aget_room(id)``), mixed in
    from the generated ``AlpakaApi``. Each generated method hands its operation
    class and variables to :meth:`execute`/:meth:`aexecute` (queries, mutations) or
    :meth:`subscribe`/:meth:`asubscribe` (subscriptions), which run it over
    ``self.rath``. Nothing is looked up; what a call returns remembers the client it
    was called on. Actions ask for it by annotation (``alpaka: Alpaka``) and are
    handed their app's client.

    The LLM tunnel is the client's too: it holds the OpenAI-compatible endpoint
    and the token loader the service built it with, and builds the ``openai``
    SDK client once, on first use of :attr:`openai` / :attr:`aopenai`.
    """


    rath: AlpakaRath = Field(
        ...,
        description="The Rath client used to interact with the Alpaka API.",
    )
    task_token: str | None = Field(
        default=None,
        description="The provenance token its requests carry; set on a per-task view only",
    )
    llm_url: str = Field(
        ..., description="The OpenAI-compatible endpoint of the alpaka server, `<alpaka>/llm/v1`"
    )
    tokens: Any = Field(
        ...,
        description=(
            "Where the tunnel's bearer token comes from, re-read on every request: a "
            "fakts TokenLoader (`aget_token()`, `arefresh_token(stale_token=None)`), "
            "typed Any because fakts is arkitekt's dependency, not this package's"
        ),
    )

    _openai: "OpenAI | None" = PrivateAttr(default=None)
    _aopenai: "AsyncOpenAI | None" = PrivateAttr(default=None)

    @property
    def openai(self) -> "OpenAI":
        """A native ``openai.OpenAI`` tunneled through this client's alpaka server.

        Built on first access and kept. Requires the ``alpaka[openai]`` extra.
        For SDK options of your own, see :func:`alpaka.tunnel.build_openai`.
        """
        if self._openai is None:
            from alpaka.tunnel import build_openai

            self._openai = build_openai(self.llm_url, self.tokens)
        return self._openai

    @property
    def aopenai(self) -> "AsyncOpenAI":
        """A native ``openai.AsyncOpenAI`` tunneled through this client's alpaka server.

        Built on first access and kept; nothing is awaited, the endpoint is a
        field. Requires the ``alpaka[openai]`` extra.
        """
        if self._aopenai is None:
            from alpaka.tunnel import build_async_openai

            self._aopenai = build_async_openai(self.llm_url, self.tokens)
        return self._aopenai

    def get_endpoint(self) -> "AlpakaEndpoint":
        """The OpenAI-compatible endpoint: base URL and a current token (it expires)."""
        from alpaka.tunnel import AlpakaEndpoint

        return AlpakaEndpoint(base_url=self.llm_url, api_key=unkoil(self.tokens.aget_token))

    async def aget_endpoint(self) -> "AlpakaEndpoint":
        """Async twin of :meth:`get_endpoint`."""
        from alpaka.tunnel import AlpakaEndpoint

        return AlpakaEndpoint(base_url=self.llm_url, api_key=await self.tokens.aget_token())

    def for_task(self, task: Any) -> "Alpaka":  # noqa: ANN401
        """A view of this client whose requests name the task they are made for.

        Shares the rath (and its connections) and the tunnel's SDK clients; only
        the token differs. rekuest hands an injected ``alpaka: Alpaka`` out through
        this.
        """
        view = self.model_copy(update={"task_token": task.token})
        view._openai = self._openai
        view._aopenai = self._aopenai
        return view

    @staticmethod
    def _serialize(operation: type[TOperation], variables: dict[str, Any]) -> dict[str, Any]:
        # alpaka omits every argument that was not set (exclude_unset), as it always has:
        # an optional left out stays off the wire rather than being sent as null.
        return operation.Arguments(**variables).model_dump(by_alias=True, exclude_unset=True)

    def _headers(self) -> dict[str, Any] | None:
        """The per-call headers: the task's provenance token, on a per-task view."""
        return {TASK_HEADER: self.task_token} if self.task_token else None

    def execute(self, operation: type[TOperation], variables: dict[str, Any]) -> TOperation:
        """Executes a query or mutation in a blocking way."""
        return unkoil(self.aexecute, operation, variables)

    async def aexecute(
        self, operation: type[TOperation], variables: dict[str, Any]
    ) -> TOperation:
        """Executes a query or mutation in a non-blocking way."""
        x = await self.rath.aquery(
            operation.Meta.document,
            self._serialize(operation, variables),
            headers=self._headers(),
        )
        return operation.model_validate(
            x.data, context=origin_context(client=self, rath=self.rath)
        )

    def subscribe(
        self, operation: type[TOperation], variables: dict[str, Any]
    ) -> Generator[TOperation, None, None]:
        """Subscribes to an operation in a blocking way."""
        return unkoil_gen(self.asubscribe, operation, variables)

    async def asubscribe(
        self, operation: type[TOperation], variables: dict[str, Any]
    ) -> AsyncGenerator[TOperation, None]:
        """Subscribes to an operation in a non-blocking way."""
        async for event in self.rath.asubscribe(
            operation.Meta.document,
            self._serialize(operation, variables),
            headers=self._headers(),
        ):
            yield operation.model_validate(
                event.data, context=origin_context(client=self, rath=self.rath)
            )
