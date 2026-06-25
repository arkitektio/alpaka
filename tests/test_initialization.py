"""Lightweight initialization tests that need no deployed stack.

These run on every OS in the CI matrix (they carry no ``integration`` marker),
exercising pure-Python construction of the client and the generated schema
models without touching the network or Docker.
"""

from alpaka.alpaka import Alpaka
from alpaka.api.schema import (
    OffsetPaginationInput,
    Ordering,
    RoomFilter,
    RoomOrder,
)


def test_alpaka_importable() -> None:
    """The top-level ``Alpaka`` composition imports and is a class."""
    assert isinstance(Alpaka, type)


def test_offset_pagination_input_constructs() -> None:
    """``OffsetPaginationInput`` builds from plain values."""
    pagination = OffsetPaginationInput(offset=0, limit=10)
    assert pagination.offset == 0
    assert pagination.limit == 10


def test_ordering_enum_values() -> None:
    """The ``Ordering`` enum exposes the expected ascending/descending members."""
    assert Ordering.ASC.value == "ASC"
    assert Ordering.DESC.value == "DESC"


def test_room_order_uses_enum() -> None:
    """``RoomOrder`` accepts an ``Ordering`` and serialises via its alias."""
    order = RoomOrder(created_at=Ordering.DESC)
    dumped = order.model_dump(by_alias=True, exclude_none=True)
    assert dumped == {"createdAt": "DESC"}


def test_room_filter_defaults_to_empty() -> None:
    """``RoomFilter`` is fully optional and constructs with no arguments."""
    assert RoomFilter().model_dump(exclude_none=True) == {}
