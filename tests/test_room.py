import pytest
from alpaka.api.schema import create_room
from .conftest import DeployedAlpaka


@pytest.mark.integration
def test_create_graph(deployed_app: DeployedAlpaka) -> None:
    """Test the creation of a graph."""
    t = create_room(
        title="Test Room",
        description="This is a test room.",
    )
    assert t.typename == "Room"
