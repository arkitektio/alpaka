import pytest
from dokker import local, Deployment
from dokker.log_watcher import LogWatcher
import os
from typing import Generator
from alpaka.alpaka import Alpaka
from rath.links.auth import ComposedAuthLink
from rath.links.aiohttp import AIOHttpLink
from rath.links.graphql_ws import GraphQLWSLink
from rath.links.timeout import TimeoutLink
from rath.links.compose import compose
from alpaka.rath import (
    AlpakaRath,
    SplitLink,
)
from graphql import OperationType
from dataclasses import dataclass

project_path = os.path.join(os.path.dirname(__file__), "integration")
docker_compose_file = os.path.join(project_path, "docker-compose.yml")


async def token_loader() -> str:
    """Load a static token. that is defined in the alpaka.yaml file. THIS IS NOT SECURE."""
    return "test"


@dataclass
class DeployedAlpaka:
    """Deployed Alpaka instance."""

    deployment: Deployment
    alpaka_watcher: LogWatcher
    alpaka: Alpaka


@pytest.fixture(scope="session")
def deployed_app() -> Generator[DeployedAlpaka, None, None]:
    """Fixture to deploy the Kraph application using Docker Compose."""
    setup = local(docker_compose_file)
    setup.add_health_check(
        url=lambda spec: f"http://localhost:{spec.find_service('alpaka').get_port_for_internal(80).published}/graphql",
        service="alpaka",
        timeout=5,
        max_retries=10,
    )

    watcher = setup.create_watcher("alpaka")

    with setup:
        http_url = f"http://localhost:{setup.spec.find_service('alpaka').get_port_for_internal(80).published}/graphql"
        ws_url = f"ws://localhost:{setup.spec.find_service('alpaka').get_port_for_internal(80).published}/graphql"

        print(f"HTTP URL: {http_url}")
        print(f"WS URL: {ws_url}")

        y = AlpakaRath(
            link=compose(
                TimeoutLink(timeout=12),
                ComposedAuthLink(
                    token_loader=token_loader, token_refresher=token_loader
                ),
                SplitLink(
                    left=AIOHttpLink(endpoint_url=http_url),
                    right=GraphQLWSLink(ws_endpoint_url=ws_url),
                    split=lambda o: o.node.operation != OperationType.SUBSCRIPTION,
                ),
            ),
        )

        alpaka = Alpaka(
            rath=y,
        )

        setup.up()

        setup.check_health()

        with alpaka as alpaka:
            deployed = DeployedAlpaka(
                deployment=setup,
                alpaka_watcher=watcher,
                alpaka=alpaka,
            )

            yield deployed
