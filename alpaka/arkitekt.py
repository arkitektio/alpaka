"""The alpaka service of an arkitekt app, and the types it sends by id.

Declared on one registry: the service first, then the structures whose expanders
ask for the client it returns. An app takes all of it in with
``App(services=[alpaka_service])``.
"""

import os
from typing import Annotated

from fakts import Alias, Require, TokenLoader
from fakts.contrib.rath.auth import FaktsAuthLink
from graphql import OperationType
from rath.links.aiohttp import AIOHttpLink
from rath.links.compose import compose
from rath.links.graphql_ws import GraphQLWSLink
from rath.links.split import SplitLink

from rekuest.app import AppRegistry
from rekuest.widgets import SearchWidget

from alpaka.api.schema import (
    LLMModel,
    Message,
    Room,
    SearchLLMModelsQuery,
    SearchMessagesQuery,
    SearchRoomsQuery,
)
from alpaka.alpaka import Alpaka
from alpaka.rath import AlpakaRath


def build_relative_path(*path: str) -> str:
    """Build a path relative to this file, for the files shipped beside it."""
    return os.path.join(os.path.dirname(__file__), *path)


registry = AppRegistry()
"""What alpaka brings to an app: its service, and the types it can send by id."""


@registry.service(
    schema=build_relative_path("api", "schema.graphql"),
    turms=build_relative_path("api", "project.json"),
)
def alpaka(
    alpaka: Annotated[
        Alias,
        Require("live.arkitekt.alpaka", "Where the models this app can prompt are served"),
    ],
    tokens: TokenLoader,
) -> Alpaka:
    """Alpaka: large language models, and the OpenAI-compatible tunnel to them.

    The resolved address serves both the GraphQL API and the ``/llm/v1`` tunnel;
    the token loader authenticates both. The client keeps them and builds the
    ``openai`` SDK client itself, lazily.
    """
    return Alpaka(
        llm_url=alpaka.to_http_path("/llm/v1"),
        tokens=tokens,
        rath=AlpakaRath(
            link=compose(
                FaktsAuthLink(token_loader=tokens),
                SplitLink(
                    left=AIOHttpLink(endpoint_url=alpaka.to_http_path("graphql")),
                    right=GraphQLWSLink(ws_endpoint_url=alpaka.to_ws_path("graphql")),
                    split=lambda o: o.node.operation != OperationType.SUBSCRIPTION,
                ),
            ),
        ),
    )


def _search(query: object) -> SearchWidget:
    """The widget that picks one of these out of the deployment."""
    return SearchWidget(query=query.Meta.document, ward="alpaka")  # type: ignore[attr-defined]


@registry.structure("@alpaka/room", widget=_search(SearchRoomsQuery))
async def expand_room(id: str, alpaka: Alpaka) -> Room:
    """A room, by id."""
    return await alpaka.aget_room(id)


@registry.structure("@alpaka/llmmodel", widget=_search(SearchLLMModelsQuery))
async def expand_llm_model(id: str, alpaka: Alpaka) -> LLMModel:
    """A model, by id."""
    return await alpaka.aget_llm_model(id)


@registry.structure("@alpaka/message", widget=_search(SearchMessagesQuery))
async def expand_message(id: str, alpaka: Alpaka) -> Message:
    """A message, by id."""
    return await alpaka.aget_message(id)
