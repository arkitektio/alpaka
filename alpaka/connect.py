"""How ``Alpaka`` reaches a resolved deployment: its links."""

from arkitekt_spec.declare.wiring import Alias, TokenLoader
from fakts.contrib.rath.auth import FaktsAuthLink
from rath.links.aiohttp import AIOHttpLink
from rath.links.compose import compose
from rath.links.graphql_ws import GraphQLWSLink
from rath.links.split import SplitLink

from alpaka.rath import AlpakaRath
from graphql import OperationType


def alpaka_rath(alpaka: Alias, tokens: TokenLoader) -> AlpakaRath:
    """The rath over http and ws."""
    return AlpakaRath(
        link=compose(
            FaktsAuthLink(token_loader=tokens),
            SplitLink(
                left=AIOHttpLink(
                    endpoint_url=alpaka.to_http_path("graphql"),
                    proxy=alpaka.proxy,
                    ssl_context=alpaka.ssl_context(),
                ),
                right=GraphQLWSLink(
                    ws_endpoint_url=alpaka.to_ws_path("graphql"),
                    proxy=alpaka.proxy,
                    ssl_context=alpaka.ssl_context(),
                ),
                split=lambda o: o.node.operation != OperationType.SUBSCRIPTION,
            ),
        ),
    )


__all__ = ["alpaka_rath"]
