"""The types alpaka sends by id: each expanded by the ``Alpaka`` client.

Imported by ``alpaka/__init__``, so each class is marked whenever alpaka is.
"""

from arkitekt_spec.declare.structures.declare import structure
from arkitekt_spec.declare.widgets import SearchWidget

from alpaka.alpaka import Alpaka
from alpaka.api.schema import (
    LLMModel,
    Message,
    Room,
    SearchLLMModelsQuery,
    SearchMessagesQuery,
    SearchRoomsQuery,
)


def _search(query: object) -> SearchWidget:
    """The widget that picks one of these out of the deployment."""
    return SearchWidget(query=query.Meta.document, ward="alpaka")  # type: ignore[attr-defined]


@structure("@alpaka/room", widget=_search(SearchRoomsQuery))
async def expand_room(id: str, alpaka: Alpaka) -> Room:
    """A room, by id."""
    return await alpaka.aget_room(id)


@structure("@alpaka/llmmodel", widget=_search(SearchLLMModelsQuery))
async def expand_llm_model(id: str, alpaka: Alpaka) -> LLMModel:
    """A model, by id."""
    return await alpaka.aget_llm_model(id)


@structure("@alpaka/message", widget=_search(SearchMessagesQuery))
async def expand_message(id: str, alpaka: Alpaka) -> Message:
    """A message, by id."""
    return await alpaka.aget_message(id)
