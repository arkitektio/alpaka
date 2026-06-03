from rekuest_next.structures.default import get_default_structure_registry, id_shrink
from rekuest_next.widgets import SearchWidget
from alpaka.api.schema import (
    Room,
    aget_room,
    LLMModel,
    Message,
    aget_message,
    SearchMessagesQuery,
    aget_llm_model,
    SearchRoomsQuery,
    SearchLLMModelsQuery,
)

structure_reg = get_default_structure_registry()
structure_reg.register_as_structure(
    Room,
    identifier="@alpaka/room",
    aexpand=aget_room,
    ashrink=id_shrink,
    default_widget=SearchWidget(query=SearchRoomsQuery.Meta.document, ward="alpaka"),
)


structure_reg.register_as_structure(
    LLMModel,
    identifier="@alpaka/llmmodel",
    aexpand=aget_llm_model,
    ashrink=id_shrink,
    default_widget=SearchWidget(
        query=SearchLLMModelsQuery.Meta.document, ward="alpaka"
    ),
)

structure_reg.register_as_structure(
    Message,
    identifier="@alpaka/message",
    aexpand=aget_message,
    ashrink=id_shrink,
    default_widget=SearchWidget(query=SearchMessagesQuery.Meta.document, ward="alpaka"),
)
