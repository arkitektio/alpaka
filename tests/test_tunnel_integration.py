"""Integration test: the OpenAI-compatible tunnel on the dokker stack.

Drives the real ``openai`` SDK against the composed alpaka server's
``/llm/v1`` endpoint with the stack's static token — the exact wiring
``alpaka.openai()`` brokers in production, minus fakts.

Marked ``xfail(strict=False)``: the pulled ``jhnnsrs/alpaka:next`` image may
predate the REST auth fix (the missing ``await`` in
``llm.views.authenticate_request``, fixed 2026-09-01), in which case every
request 401s. Once the image is rebuilt this xpasses; drop the marker then.
"""

import pytest

from .conftest import DeployedAlpaka

openai_sdk = pytest.importorskip("openai")


def _tunnel_base_url(deployed_app: DeployedAlpaka) -> str:
    port = (
        deployed_app.deployment.spec.find_service("alpaka")
        .get_port_for_internal(80)
        .published
    )
    return f"http://localhost:{port}/llm/v1"


@pytest.mark.integration
@pytest.mark.xfail(
    strict=False,
    reason="pulled alpaka image may predate the REST auth await fix (2026-09-01)",
)
def test_models_endpoint_speaks_openai_wire(deployed_app: DeployedAlpaka) -> None:
    """``client.models.list()`` authenticates with the static token and parses."""
    client = openai_sdk.OpenAI(
        base_url=_tunnel_base_url(deployed_app), api_key="test"
    )
    models = client.models.list()
    assert models.object == "list"


@pytest.mark.integration
def test_tunnel_rejects_bad_token(deployed_app: DeployedAlpaka) -> None:
    """A wrong token must 401 — on any image version, buggy or fixed."""
    client = openai_sdk.OpenAI(
        base_url=_tunnel_base_url(deployed_app), api_key="not-a-valid-token"
    )
    with pytest.raises(openai_sdk.AuthenticationError):
        client.models.list()
