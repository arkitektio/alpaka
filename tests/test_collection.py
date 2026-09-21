import pytest

from alpaka.alpaka import Alpaka
from alpaka.api.schema import (
    ChromaCollectionFilter,
    ChromaCollectionOrderCreatedAt,
    ChromaCollectionOrderName,
    OffsetPaginationInput,
    Ordering,
)


@pytest.mark.integration
def test_list_chroma_collections_accepts_ordering(
    alpaka: Alpaka,
) -> None:
    """Listing collections accepts an ordering argument and returns a list."""
    collections = alpaka.list_chroma_collections(
        order=[ChromaCollectionOrderName(name=Ordering.ASC)],
        pagination=OffsetPaginationInput(limit=100),
    )
    assert isinstance(collections, tuple)


@pytest.mark.integration
def test_list_chroma_collections_accepts_filter_and_ordering(
    alpaka: Alpaka,
) -> None:
    """Collections can be filtered and ordered in the same query."""
    collections = alpaka.list_chroma_collections(
        filter=ChromaCollectionFilter(search="anything"),
        order=[ChromaCollectionOrderCreatedAt(createdAt=Ordering.DESC)],
    )
    assert isinstance(collections, tuple)


@pytest.mark.integration
def test_search_chroma_collection_respects_limit(
    alpaka: Alpaka,
) -> None:
    """The collection search honours the limit argument."""
    options = alpaka.search_chroma_collection(search="anything", limit=2, offset=0)
    assert len(options) <= 2
