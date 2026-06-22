import pytest
from alpaka.api.schema import (
    list_chroma_collections,
    search_chroma_collection,
    ChromaCollectionFilter,
    ChromaCollectionOrder,
    OffsetPaginationInput,
    Ordering,
)
from .conftest import DeployedAlpaka


@pytest.mark.integration
def test_list_chroma_collections_accepts_ordering(
    deployed_app: DeployedAlpaka,
) -> None:
    """Listing collections accepts an ordering argument and returns a list."""
    collections = list_chroma_collections(
        order=[ChromaCollectionOrder(name=Ordering.ASC)],
        pagination=OffsetPaginationInput(limit=100),
    )
    assert isinstance(collections, tuple)


@pytest.mark.integration
def test_list_chroma_collections_accepts_filter_and_ordering(
    deployed_app: DeployedAlpaka,
) -> None:
    """Collections can be filtered and ordered in the same query."""
    collections = list_chroma_collections(
        filter=ChromaCollectionFilter(search="anything"),
        order=[ChromaCollectionOrder(createdAt=Ordering.DESC)],
    )
    assert isinstance(collections, tuple)


@pytest.mark.integration
def test_search_chroma_collection_respects_limit(
    deployed_app: DeployedAlpaka,
) -> None:
    """The collection search honours the limit argument."""
    options = search_chroma_collection(search="anything", limit=2, offset=0)
    assert len(options) <= 2
