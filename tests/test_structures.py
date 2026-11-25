"""alpaka's structures: what it can send across a node boundary, and how.

Nothing here needs a server. Each type is declared once, with the expander that
fetches it back, so there is no second list to drift from -- which is what the
old ``EXPANDERS``-versus-``STRUCTURES`` test existed to catch.
"""

import pytest

pytest.importorskip("rekuest")

from arkitekt_spec.declare.client import client_of  # noqa: E402
from arkitekt_spec.declare.structures.types import is_valid_identifier  # noqa: E402

from alpaka import Alpaka  # noqa: E402

DECLARED = client_of(Alpaka)
assert DECLARED is not None
STRUCTURES = {s.identifier: s for s in DECLARED.structures}

#: The identifiers alpaka puts on the wire. Changing one is a cross-service change,
#: so they are written out rather than derived from the thing under test.
IDENTIFIERS = {
    "@alpaka/llmmodel",
    "@alpaka/message",
    "@alpaka/room",
}


def test_it_declares_exactly_the_identifiers_it_always_has() -> None:
    """The wire contract, guarded literally."""
    assert set(STRUCTURES) == IDENTIFIERS


def test_every_identifier_is_one_the_server_accepts() -> None:
    assert [i for i in IDENTIFIERS if not is_valid_identifier(i)] == []


def test_every_structure_is_expanded_by_alpaka() -> None:
    """An expander, owned by the client it asks for."""
    for structure in STRUCTURES.values():
        assert structure.aexpand is not None, structure.identifier
        assert structure.owner is DECLARED, structure.identifier


def test_importing_alpaka_marks_its_structures() -> None:
    """A plain import of the schema marks the classes: nothing else has to be imported."""
    import subprocess
    import sys

    script = (
        "from arkitekt_spec.declare.structures.declare import structure_of\n"
        "from alpaka.api.schema import Room\n"
        "assert structure_of(Room).owner.cls.__name__ == 'Alpaka'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_every_structure_class_the_server_identifies_says_so_itself() -> None:
    """`__identifier__` (turms over `@identified`) agrees with the identifier each class is registered under."""
    stamped = {
        identifier: getattr(structure.cls, "__identifier__", None)
        for identifier, structure in STRUCTURES.items()
    }
    assert {i: s for i, s in stamped.items() if s is not None} == {
        i: i for i, s in stamped.items() if s is not None
    }
    assert any(s is not None for s in stamped.values())
