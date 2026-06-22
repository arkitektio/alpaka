import pytest
from alpaka.api.schema import (
    list_ll_models,
    search_llm_models,
    LLMModelFilter,
    LLMModelOrder,
    OffsetPaginationInput,
    Ordering,
)
from .conftest import DeployedAlpaka


@pytest.mark.integration
def test_list_ll_models_accepts_ordering(deployed_app: DeployedAlpaka) -> None:
    """Listing models accepts an ordering argument and returns a list."""
    models = list_ll_models(
        order=[LLMModelOrder(label=Ordering.ASC)],
        pagination=OffsetPaginationInput(limit=100),
    )
    assert isinstance(models, tuple)


@pytest.mark.integration
def test_list_ll_models_accepts_filter_and_ordering(
    deployed_app: DeployedAlpaka,
) -> None:
    """Models can be filtered and ordered in the same query."""
    models = list_ll_models(
        filter=LLMModelFilter(search="anything"),
        order=[LLMModelOrder(modelId=Ordering.DESC)],
    )
    assert isinstance(models, tuple)


@pytest.mark.integration
def test_search_llm_models_respects_limit(deployed_app: DeployedAlpaka) -> None:
    """The model search honours the limit argument."""
    options = search_llm_models(search="anything", limit=2, offset=0)
    assert len(options) <= 2
