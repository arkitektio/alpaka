import pytest

from alpaka.alpaka import Alpaka
from alpaka.api.schema import (
    LLMModelFilter,
    LLMModelOrderLabel,
    LLMModelOrderModelId,
    OffsetPaginationInput,
    Ordering,
)


@pytest.mark.integration
def test_list_ll_models_accepts_ordering(alpaka: Alpaka) -> None:
    """Listing models accepts an ordering argument and returns a list."""
    models = alpaka.list_ll_models(
        order=[LLMModelOrderLabel(label=Ordering.ASC)],
        pagination=OffsetPaginationInput(limit=100),
    )
    assert isinstance(models, tuple)


@pytest.mark.integration
def test_list_ll_models_accepts_filter_and_ordering(
    alpaka: Alpaka,
) -> None:
    """Models can be filtered and ordered in the same query."""
    models = alpaka.list_ll_models(
        filter=LLMModelFilter(search="anything"),
        order=[LLMModelOrderModelId(modelId=Ordering.DESC)],
    )
    assert isinstance(models, tuple)


@pytest.mark.integration
def test_search_llm_models_respects_limit(alpaka: Alpaka) -> None:
    """The model search honours the limit argument."""
    options = alpaka.search_llm_models(search="anything", limit=2, offset=0)
    assert len(options) <= 2
