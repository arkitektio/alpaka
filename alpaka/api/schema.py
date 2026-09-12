from alpaka.funcs import aexecute, asubscribe, execute, subscribe
from alpaka.rath import AlpakaRath
from alpaka.traits import ChatResponseTraits
from datetime import datetime
from enum import Enum
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from rath.scalars import ID, IDCoercible
from typing import Annotated, Any, AsyncIterator, Iterable, Iterator, Literal

class GraphQLDefault:
    """Records a GraphQL field schema default value. The client omits the field so the server applies its own default; this preserves the value for introspection."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'GraphQLDefault(' + repr(self.value) + ')'

class UnsetType:
    """Sentinel for arguments the caller did not provide. Such fields are omitted on serialization so the GraphQL server applies its own default."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False
UNSET = UnsetType()

class FeatureType(str, Enum):
    """A capability a model supports"""
    EMBEDDING = 'EMBEDDING'
    CHAT = 'CHAT'
    VISION = 'VISION'
    __str__ = str.__str__

class Modality(str, Enum):
    """A modality a model can read or emit"""
    IMAGE = 'IMAGE'
    TEXT = 'TEXT'
    AUDIO = 'AUDIO'
    VIDEO = 'VIDEO'
    __str__ = str.__str__

class Ordering(str, Enum):
    """No documentation"""
    ASC = 'ASC'
    ASC_NULLS_FIRST = 'ASC_NULLS_FIRST'
    ASC_NULLS_LAST = 'ASC_NULLS_LAST'
    DESC = 'DESC'
    DESC_NULLS_FIRST = 'DESC_NULLS_FIRST'
    DESC_NULLS_LAST = 'DESC_NULLS_LAST'
    __str__ = str.__str__

class ProviderKind(str, Enum):
    """The kind of LLM provider"""
    OPENAI = 'OPENAI'
    ANTHROPIC = 'ANTHROPIC'
    GOOGLE = 'GOOGLE'
    COHERE = 'COHERE'
    MISTRAL = 'MISTRAL'
    HUGGINGFACE = 'HUGGINGFACE'
    OLLAMA = 'OLLAMA'
    AZURE = 'AZURE'
    AWS = 'AWS'
    VERTEX_AI = 'VERTEX_AI'
    PALM = 'PALM'
    REPLICATE = 'REPLICATE'
    TOGETHER_AI = 'TOGETHER_AI'
    ANYSCALE = 'ANYSCALE'
    FIREWORKS_AI = 'FIREWORKS_AI'
    DEEPINFRA = 'DEEPINFRA'
    PERPLEXITY = 'PERPLEXITY'
    GROQ = 'GROQ'
    CUSTOM = 'CUSTOM'
    UNKNOWN = 'UNKNOWN'
    OPENROUTER = 'OPENROUTER'
    __str__ = str.__str__

class Role(str, Enum):
    """The type of the message sender"""
    SYSTEM = 'SYSTEM'
    USER = 'USER'
    ASSISTANT = 'ASSISTANT'
    TOOL = 'TOOL'
    FUNCTION = 'FUNCTION'
    __str__ = str.__str__

class RoomEventKind(str, Enum):
    """What happened in a room"""
    MESSAGE_CREATED = 'MESSAGE_CREATED'
    MESSAGE_UPDATED = 'MESSAGE_UPDATED'
    MESSAGE_FINISHED = 'MESSAGE_FINISHED'
    JOIN = 'JOIN'
    LEAVE = 'LEAVE'
    __str__ = str.__str__

class ThinkingBlockType(str, Enum):
    """The type of the thinking block"""
    THINKING = 'THINKING'
    __str__ = str.__str__

class ToolType(str, Enum):
    """The type of the tool"""
    FUNCTION = 'FUNCTION'
    __str__ = str.__str__

class AddDocumentsToCollectionInput(BaseModel):
    """Documents to add to an existing collection"""
    collection: ID
    documents: tuple['DocumentInput', ...]
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChatInput(BaseModel):
    """A chat completion request"""
    messages: tuple['ChatMessageInput', ...]
    model: ID | None = None
    tools: tuple['ToolInput', ...] | None = None
    tool_choice: Any | None = Field(validation_alias=AliasChoices('tool_choice', 'toolChoice'), serialization_alias='toolChoice', default=None)
    temperature: float | None = None
    max_tokens: int | None = Field(validation_alias=AliasChoices('max_tokens', 'maxTokens'), serialization_alias='maxTokens', default=None)
    top_p: float | None = Field(validation_alias=AliasChoices('top_p', 'topP'), serialization_alias='topP', default=None)
    frequency_penalty: float | None = Field(validation_alias=AliasChoices('frequency_penalty', 'frequencyPenalty'), serialization_alias='frequencyPenalty', default=None)
    presence_penalty: float | None = Field(validation_alias=AliasChoices('presence_penalty', 'presencePenalty'), serialization_alias='presencePenalty', default=None)
    stop: tuple[str, ...] | None = None
    n: int | None = None
    response_format: Any | None = Field(validation_alias=AliasChoices('response_format', 'responseFormat'), serialization_alias='responseFormat', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChatMessageInput(BaseModel):
    """A chat message input"""
    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = Field(validation_alias=AliasChoices('tool_call_id', 'toolCallId'), serialization_alias='toolCallId', default=None)
    function_call: 'FunctionCallInput | None' = Field(validation_alias=AliasChoices('function_call', 'functionCall'), serialization_alias='functionCall', default=None)
    tool_calls: tuple['ToolCallInput', ...] | None = Field(validation_alias=AliasChoices('tool_calls', 'toolCalls'), serialization_alias='toolCalls', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionFilter(BaseModel):
    """Filter for ChromaCollection"""
    and_: 'ChromaCollectionFilter | None' = Field(validation_alias=AliasChoices('and_', 'AND'), serialization_alias='AND', default=None)
    or_: 'ChromaCollectionFilter | None' = Field(validation_alias=AliasChoices('or_', 'OR'), serialization_alias='OR', default=None)
    not_: 'ChromaCollectionFilter | None' = Field(validation_alias=AliasChoices('not_', 'NOT'), serialization_alias='NOT', default=None)
    distinct: bool | None = Field(validation_alias=AliasChoices('distinct', 'DISTINCT'), serialization_alias='DISTINCT', default=None)
    ids: tuple[ID, ...] | None = None
    search: str | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionInput(BaseModel):
    """A collection of documents searchable by string"""
    name: str
    embedder: ID
    description: str | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionOrderName(BaseModel):
    """'name' variant of the @oneOf input 'ChromaCollectionOrder'"""
    name: Ordering
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionOrderCreatedAt(BaseModel):
    """'createdAt' variant of the @oneOf input 'ChromaCollectionOrder'"""
    created_at: Ordering = Field(validation_alias=AliasChoices('created_at', 'createdAt'), serialization_alias='createdAt')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)
ChromaCollectionOrder = ChromaCollectionOrderName | ChromaCollectionOrderCreatedAt

class DocumentInput(BaseModel):
    """A document to put into the vector database"""
    content: str
    structure: 'StructureInput | None' = None
    id: str | None = None
    metadata: Any | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class FunctionCallInput(BaseModel):
    """A function call input"""
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class FunctionDefinitionInput(BaseModel):
    """A large language model function defintion"""
    name: str
    description: str | None = None
    parameters: Any | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ImageInput(BaseModel):
    """The image"""
    model: ID | None = None
    description: str
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class LLMModelFilter(BaseModel):
    """Filter for LLMModel"""
    and_: 'LLMModelFilter | None' = Field(validation_alias=AliasChoices('and_', 'AND'), serialization_alias='AND', default=None)
    or_: 'LLMModelFilter | None' = Field(validation_alias=AliasChoices('or_', 'OR'), serialization_alias='OR', default=None)
    not_: 'LLMModelFilter | None' = Field(validation_alias=AliasChoices('not_', 'NOT'), serialization_alias='NOT', default=None)
    distinct: bool | None = Field(validation_alias=AliasChoices('distinct', 'DISTINCT'), serialization_alias='DISTINCT', default=None)
    ids: tuple[ID, ...] | None = None
    search: str | None = None
    input_modalities: tuple[Modality, ...] | None = Field(validation_alias=AliasChoices('input_modalities', 'inputModalities'), serialization_alias='inputModalities', default=None)
    output_modalities: tuple[Modality, ...] | None = Field(validation_alias=AliasChoices('output_modalities', 'outputModalities'), serialization_alias='outputModalities', default=None)
    features: tuple[FeatureType, ...] | None = None
    provider: ID | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class LLMModelOrderLabel(BaseModel):
    """'label' variant of the @oneOf input 'LLMModelOrder'"""
    label: Ordering
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class LLMModelOrderModelId(BaseModel):
    """'modelId' variant of the @oneOf input 'LLMModelOrder'"""
    model_id: Ordering = Field(validation_alias=AliasChoices('model_id', 'modelId'), serialization_alias='modelId')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)
LLMModelOrder = LLMModelOrderLabel | LLMModelOrderModelId

class MessageFilter(BaseModel):
    """Message represent the message of an agent on a room"""
    and_: 'MessageFilter | None' = Field(validation_alias=AliasChoices('and_', 'AND'), serialization_alias='AND', default=None)
    or_: 'MessageFilter | None' = Field(validation_alias=AliasChoices('or_', 'OR'), serialization_alias='OR', default=None)
    not_: 'MessageFilter | None' = Field(validation_alias=AliasChoices('not_', 'NOT'), serialization_alias='NOT', default=None)
    distinct: bool | None = Field(validation_alias=AliasChoices('distinct', 'DISTINCT'), serialization_alias='DISTINCT', default=None)
    ids: tuple[ID, ...] | None = None
    search: str | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class MessageOrderCreatedAt(BaseModel):
    """'createdAt' variant of the @oneOf input 'MessageOrder'"""
    created_at: Ordering = Field(validation_alias=AliasChoices('created_at', 'createdAt'), serialization_alias='createdAt')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)
MessageOrder = MessageOrderCreatedAt

class OffsetPaginationInput(BaseModel):
    """No documentation"""
    offset: Annotated[int | None, GraphQLDefault('0')] = None
    'Default: 0'
    limit: int | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ProviderInput(BaseModel):
    """A large language model to change with"""
    description: str | None = None
    name: str | None = None
    kind: ProviderKind
    api_key: str | None = Field(validation_alias=AliasChoices('api_key', 'apiKey'), serialization_alias='apiKey', default=None)
    api_base: str | None = Field(validation_alias=AliasChoices('api_base', 'apiBase'), serialization_alias='apiBase', default=None)
    additional_config: Any | None = Field(validation_alias=AliasChoices('additional_config', 'additionalConfig'), serialization_alias='additionalConfig', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class PullInput(BaseModel):
    """The model to pull, and the provider to pull it into"""
    model_name: str = Field(validation_alias=AliasChoices('model_name', 'modelName'), serialization_alias='modelName')
    provider: ID | None = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class QueryInput(BaseModel):
    """A similarity query against a collection"""
    collection: ID
    query_texts: tuple[str, ...] = Field(validation_alias=AliasChoices('query_texts', 'queryTexts'), serialization_alias='queryTexts', description='One or more query texts; the union of their results is returned, deduplicated by document')
    n_results: Annotated[int | None, GraphQLDefault('3')] = Field(validation_alias=AliasChoices('n_results', 'nResults'), serialization_alias='nResults', default=None, description='Results per query text')
    'Results per query text\nDefault: 3'
    where: Any | None = Field(default=None, description='Chroma metadata filter applied to every query')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class RoomFilter(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    and_: 'RoomFilter | None' = Field(validation_alias=AliasChoices('and_', 'AND'), serialization_alias='AND', default=None)
    or_: 'RoomFilter | None' = Field(validation_alias=AliasChoices('or_', 'OR'), serialization_alias='OR', default=None)
    not_: 'RoomFilter | None' = Field(validation_alias=AliasChoices('not_', 'NOT'), serialization_alias='NOT', default=None)
    distinct: bool | None = Field(validation_alias=AliasChoices('distinct', 'DISTINCT'), serialization_alias='DISTINCT', default=None)
    ids: tuple[ID, ...] | None = None
    search: str | None = None
    talking_about: 'StructureInput | None' = Field(validation_alias=AliasChoices('talking_about', 'talkingAbout'), serialization_alias='talkingAbout', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class RoomOrderCreatedAt(BaseModel):
    """'createdAt' variant of the @oneOf input 'RoomOrder'"""
    created_at: Ordering = Field(validation_alias=AliasChoices('created_at', 'createdAt'), serialization_alias='createdAt')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class RoomOrderTitle(BaseModel):
    """'title' variant of the @oneOf input 'RoomOrder'"""
    title: Ordering
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)
RoomOrder = RoomOrderCreatedAt | RoomOrderTitle

class StructureInput(BaseModel):
    """A reference to an object held by another Arkitekt service"""
    identifier: str
    object: int
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ToolCallInput(BaseModel):
    """A tool call input"""
    id: str
    function: FunctionCallInput
    type: ToolType
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ToolInput(BaseModel):
    """A large language model function call"""
    type: Annotated[ToolType | None, GraphQLDefault('FUNCTION')] = None
    'Default: FUNCTION'
    function: FunctionDefinitionInput
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollection(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    id: ID
    name: str
    'The human-readable name of the collection, unique within its organization'
    description: str
    created_at: datetime = Field(alias='createdAt')
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ChromaCollection"""
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}'
        name = 'ChromaCollection'
        type = 'ChromaCollection'

class Document(BaseModel):
    """A document stored in a collection"""
    typename: Literal['Document'] = Field(alias='__typename', default='Document', exclude=True)
    id: str
    content: str
    metadata: Any | None = Field(default=None)
    'The metadata stored alongside the document'
    distance: float | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Document"""
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}'
        name = 'Document'
        type = 'Document'

class LLMModelProvider(BaseModel):
    """A provider of LLMs"""
    typename: Literal['Provider'] = Field(alias='__typename', default='Provider', exclude=True)
    id: ID
    name: str
    model_config = ConfigDict(frozen=True)

class LLMModelEmbedderFor(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    id: ID
    name: str
    'The human-readable name of the collection, unique within its organization'
    model_config = ConfigDict(frozen=True)

class LLMModel(BaseModel):
    """A LLM model to chage with"""
    typename: Literal['LLMModel'] = Field(alias='__typename', default='LLMModel', exclude=True)
    id: ID
    provider: LLMModelProvider
    features: tuple[FeatureType, ...]
    'The features supported by the model'
    embedder_for: tuple[LLMModelEmbedderFor, ...] = Field(alias='embedderFor')
    'The collections that can be embedded with this model'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for LLMModel"""
        document = 'fragment LLMModel on LLMModel {\n  id\n  provider {\n    id\n    name\n    __typename\n  }\n  features\n  embedderFor {\n    id\n    name\n    __typename\n  }\n  __typename\n}'
        name = 'LLMModel'
        type = 'LLMModel'

class ListMessageAgent(BaseModel):
    """A participant in a room"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    name: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class ListMessageAttachedStructures(BaseModel):
    """A reference to an object held by another Arkitekt service"""
    typename: Literal['Structure'] = Field(alias='__typename', default='Structure', exclude=True)
    object: int
    identifier: str
    model_config = ConfigDict(frozen=True)

class ListMessage(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    id: ID
    text: str
    'A clear text representation of the rich comment'
    is_streaming: bool = Field(alias='isStreaming')
    'Whether this message is still being written'
    agent: ListMessageAgent
    'The user that created this comment'
    attached_structures: tuple[ListMessageAttachedStructures, ...] = Field(alias='attachedStructures')
    'The objects this message was posted about'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ListMessage"""
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}'
        name = 'ListMessage'
        type = 'Message'

class ProviderModels(BaseModel):
    """A LLM model to chage with"""
    typename: Literal['LLMModel'] = Field(alias='__typename', default='LLMModel', exclude=True)
    id: ID
    model_id: str = Field(alias='modelId')
    features: tuple[FeatureType, ...]
    'The features supported by the model'
    model_config = ConfigDict(frozen=True)

class Provider(BaseModel):
    """A provider of LLMs"""
    typename: Literal['Provider'] = Field(alias='__typename', default='Provider', exclude=True)
    id: ID
    name: str
    models: tuple[ProviderModels, ...]
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Provider"""
        document = 'fragment Provider on Provider {\n  id\n  name\n  models {\n    id\n    modelId\n    features\n    __typename\n  }\n  __typename\n}'
        name = 'Provider'
        type = 'Provider'

class ChatResponseUsage(BaseModel):
    """No documentation"""
    typename: Literal['Usage'] = Field(alias='__typename', default='Usage', exclude=True)
    prompt_tokens: int = Field(alias='promptTokens')
    completion_tokens: int = Field(alias='completionTokens')
    total_tokens: int = Field(alias='totalTokens')
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesThinkingBlocks(BaseModel):
    """No documentation"""
    typename: Literal['ThinkingBlock'] = Field(alias='__typename', default='ThinkingBlock', exclude=True)
    type: ThinkingBlockType
    thinking: str
    signature: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessageFunctionCall(BaseModel):
    """The type of the tool"""
    typename: Literal['FunctionCall'] = Field(alias='__typename', default='FunctionCall', exclude=True)
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessageToolCallsFunction(BaseModel):
    """The type of the tool"""
    typename: Literal['FunctionCall'] = Field(alias='__typename', default='FunctionCall', exclude=True)
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessageToolCalls(BaseModel):
    """A function definition for a large language model"""
    typename: Literal['ToolCall'] = Field(alias='__typename', default='ToolCall', exclude=True)
    id: str
    type: ToolType
    function: ChatResponseChoicesMessageToolCallsFunction
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessage(BaseModel):
    """No documentation"""
    typename: Literal['ChatMessage'] = Field(alias='__typename', default='ChatMessage', exclude=True)
    role: Role
    content: str | None = Field(default=None)
    name: str | None = Field(default=None)
    tool_call_id: str | None = Field(default=None, alias='toolCallId')
    function_call: ChatResponseChoicesMessageFunctionCall | None = Field(default=None, alias='functionCall')
    tool_calls: tuple[ChatResponseChoicesMessageToolCalls, ...] | None = Field(default=None, alias='toolCalls')
    model_config = ConfigDict(frozen=True)

class ChatResponseChoices(BaseModel):
    """No documentation"""
    typename: Literal['Choice'] = Field(alias='__typename', default='Choice', exclude=True)
    index: int
    finish_reason: str | None = Field(default=None, alias='finishReason')
    reasoning_content: str | None = Field(default=None, alias='reasoningContent')
    thinking_blocks: tuple[ChatResponseChoicesThinkingBlocks, ...] | None = Field(default=None, alias='thinkingBlocks')
    message: ChatResponseChoicesMessage
    model_config = ConfigDict(frozen=True)

class ChatResponse(ChatResponseTraits, BaseModel):
    """No documentation"""
    typename: Literal['ChatResponse'] = Field(alias='__typename', default='ChatResponse', exclude=True)
    id: str
    object: str
    created: int
    model: str
    usage: ChatResponseUsage | None = Field(default=None)
    choices: tuple[ChatResponseChoices, ...]
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ChatResponse"""
        document = 'fragment ChatResponse on ChatResponse {\n  id\n  object\n  created\n  model\n  usage {\n    promptTokens\n    completionTokens\n    totalTokens\n    __typename\n  }\n  choices {\n    index\n    finishReason\n    reasoningContent\n    thinkingBlocks {\n      type\n      thinking\n      signature\n      __typename\n    }\n    message {\n      role\n      content\n      name\n      toolCallId\n      functionCall {\n        name\n        arguments\n        __typename\n      }\n      toolCalls {\n        id\n        type\n        function {\n          name\n          arguments\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  __typename\n}'
        name = 'ChatResponse'
        type = 'ChatResponse'

class Room(BaseModel):
    """A room agents and users converse in"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    title: str
    'The Title of the Room'
    description: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Room"""
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}'
        name = 'Room'
        type = 'Room'

class MessageAgentRoom(BaseModel):
    """A room agents and users converse in"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    model_config = ConfigDict(frozen=True)

class MessageAgent(BaseModel):
    """A participant in a room"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    name: str | None = Field(default=None)
    room: MessageAgentRoom
    model_config = ConfigDict(frozen=True)

class MessageRoom(BaseModel):
    """A room agents and users converse in"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    title: str
    'The Title of the Room'
    model_config = ConfigDict(frozen=True)

class MessageAttachedStructures(BaseModel):
    """A reference to an object held by another Arkitekt service"""
    typename: Literal['Structure'] = Field(alias='__typename', default='Structure', exclude=True)
    object: int
    identifier: str
    model_config = ConfigDict(frozen=True)

class Message(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    id: ID
    text: str
    'A clear text representation of the rich comment'
    is_streaming: bool = Field(alias='isStreaming')
    'Whether this message is still being written'
    agent: MessageAgent
    'The user that created this comment'
    room: MessageRoom
    before: tuple[ListMessage, ...]
    attached_structures: tuple[MessageAttachedStructures, ...] = Field(alias='attachedStructures')
    'The objects this message was posted about'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Message"""
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}'
        name = 'Message'
        type = 'Message'

class ChatMutation(BaseModel):
    """No documentation found for this operation."""
    chat: ChatResponse
    'Send a chat completion request'

    class Arguments(BaseModel):
        """Arguments for Chat """
        input: ChatInput

    class Meta:
        """Meta class for Chat """
        document = 'fragment ChatResponse on ChatResponse {\n  id\n  object\n  created\n  model\n  usage {\n    promptTokens\n    completionTokens\n    totalTokens\n    __typename\n  }\n  choices {\n    index\n    finishReason\n    reasoningContent\n    thinkingBlocks {\n      type\n      thinking\n      signature\n      __typename\n    }\n    message {\n      role\n      content\n      name\n      toolCallId\n      functionCall {\n        name\n        arguments\n        __typename\n      }\n      toolCalls {\n        id\n        type\n        function {\n          name\n          arguments\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nmutation Chat($input: ChatInput!) {\n  chat(input: $input) {\n    ...ChatResponse\n    __typename\n  }\n}'

class CreateCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    create_collection: ChromaCollection = Field(alias='createCollection')
    'Create a searchable collection of documents'

    class Arguments(BaseModel):
        """Arguments for CreateCollection """
        input: ChromaCollectionInput

    class Meta:
        """Meta class for CreateCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nmutation CreateCollection($input: ChromaCollectionInput!) {\n  createCollection(input: $input) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class EnsureCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    ensure_collection: ChromaCollection = Field(alias='ensureCollection')
    'Create a collection, or update it if it already exists'

    class Arguments(BaseModel):
        """Arguments for EnsureCollection """
        input: ChromaCollectionInput

    class Meta:
        """Meta class for EnsureCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nmutation EnsureCollection($input: ChromaCollectionInput!) {\n  ensureCollection(input: $input) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class AddDocumentsToCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    add_documents_to_collection: tuple[Document, ...] = Field(alias='addDocumentsToCollection')
    'Embed documents and add them to a collection'

    class Arguments(BaseModel):
        """Arguments for AddDocumentsToCollection """
        input: AddDocumentsToCollectionInput

    class Meta:
        """Meta class for AddDocumentsToCollection """
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}\n\nmutation AddDocumentsToCollection($input: AddDocumentsToCollectionInput!) {\n  addDocumentsToCollection(input: $input) {\n    ...Document\n    __typename\n  }\n}'

class GenerateImageMutationGenerateImage(BaseModel):
    """A generated image, base64 encoded"""
    typename: Literal['ImageResponse'] = Field(alias='__typename', default='ImageResponse', exclude=True)
    image: str
    model_config = ConfigDict(frozen=True)

class GenerateImageMutation(BaseModel):
    """No documentation found for this operation."""
    generate_image: GenerateImageMutationGenerateImage = Field(alias='generateImage')
    'Generate an image from a text description'

    class Arguments(BaseModel):
        """Arguments for GenerateImage """
        input: ImageInput

    class Meta:
        """Meta class for GenerateImage """
        document = 'mutation GenerateImage($input: ImageInput!) {\n  generateImage(input: $input) {\n    image\n    __typename\n  }\n}'

class SendMutation(BaseModel):
    """No documentation found for this operation."""
    send: Message
    'Post a complete message into a room'

    class Arguments(BaseModel):
        """Arguments for Send """
        text: str
        room: ID
        agent_id: str = Field(validation_alias=AliasChoices('agent_id', 'agentId'), serialization_alias='agentId')
        attach_structures: list[StructureInput] | None = Field(validation_alias=AliasChoices('attach_structures', 'attachStructures'), serialization_alias='attachStructures', default=None)

    class Meta:
        """Meta class for Send """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nmutation Send($text: String!, $room: ID!, $agentId: String!, $attachStructures: [StructureInput!]) {\n  send(\n    input: {text: $text, room: $room, agentId: $agentId, attachStructures: $attachStructures}\n  ) {\n    ...Message\n    __typename\n  }\n}'

class CreateProviderMutation(BaseModel):
    """No documentation found for this operation."""
    create_provider: Provider = Field(alias='createProvider')
    'Configure a new LLM provider and list the models it offers'

    class Arguments(BaseModel):
        """Arguments for CreateProvider """
        input: ProviderInput

    class Meta:
        """Meta class for CreateProvider """
        document = 'fragment Provider on Provider {\n  id\n  name\n  models {\n    id\n    modelId\n    features\n    __typename\n  }\n  __typename\n}\n\nmutation CreateProvider($input: ProviderInput!) {\n  createProvider(input: $input) {\n    ...Provider\n    __typename\n  }\n}'

class PullMutationPull(BaseModel):
    """The outcome of pulling a model"""
    typename: Literal['OllamaPullResult'] = Field(alias='__typename', default='OllamaPullResult', exclude=True)
    status: str
    detail: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class PullMutation(BaseModel):
    """No documentation found for this operation."""
    pull: PullMutationPull
    'Pull a model into an Ollama provider'

    class Arguments(BaseModel):
        """Arguments for Pull """
        input: PullInput

    class Meta:
        """Meta class for Pull """
        document = 'mutation Pull($input: PullInput!) {\n  pull(input: $input) {\n    status\n    detail\n    __typename\n  }\n}'

class CreateRoomMutation(BaseModel):
    """No documentation found for this operation."""
    create_room: Room = Field(alias='createRoom')
    'Open a new room'

    class Arguments(BaseModel):
        """Arguments for CreateRoom """
        title: str | None = Field(default=None)
        description: str | None = Field(default=None)

    class Meta:
        """Meta class for CreateRoom """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nmutation CreateRoom($title: String, $description: String) {\n  createRoom(input: {title: $title, description: $description}) {\n    ...Room\n    __typename\n  }\n}'

class StartMessageMutation(BaseModel):
    """No documentation found for this operation."""
    start_message: ListMessage = Field(alias='startMessage')
    'Open a message to stream text into. Only the starting agent (same user and client) can append to or finish it.'

    class Arguments(BaseModel):
        """Arguments for StartMessage """
        room: ID
        agent_id: str = Field(validation_alias=AliasChoices('agent_id', 'agentId'), serialization_alias='agentId')
        parent: ID | None = Field(default=None)
        text: str | None = Field(default=None)

    class Meta:
        """Meta class for StartMessage """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nmutation StartMessage($room: ID!, $agentId: String!, $parent: ID, $text: String) {\n  startMessage(\n    input: {room: $room, agentId: $agentId, parent: $parent, text: $text}\n  ) {\n    ...ListMessage\n    __typename\n  }\n}'

class AppendMessageMutation(BaseModel):
    """No documentation found for this operation."""
    append_message: ListMessage = Field(alias='appendMessage')
    'Append a delta to a streaming message. Batch deltas (every ~100-250 ms or ~30 characters) and await each call before sending the next, so they arrive in order.'

    class Arguments(BaseModel):
        """Arguments for AppendMessage """
        message: ID
        delta: str

    class Meta:
        """Meta class for AppendMessage """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nmutation AppendMessage($message: ID!, $delta: String!) {\n  appendMessage(input: {message: $message, delta: $delta}) {\n    ...ListMessage\n    __typename\n  }\n}'

class FinishMessageMutation(BaseModel):
    """No documentation found for this operation."""
    finish_message: ListMessage = Field(alias='finishMessage')
    'Close a streaming message. Pass the full final text so any delta lost on the way is repaired; call it in a finally block so a crashed stream never stays open.'

    class Arguments(BaseModel):
        """Arguments for FinishMessage """
        message: ID
        text: str | None = Field(default=None)
        attach_structures: list[StructureInput] | None = Field(validation_alias=AliasChoices('attach_structures', 'attachStructures'), serialization_alias='attachStructures', default=None)

    class Meta:
        """Meta class for FinishMessage """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nmutation FinishMessage($message: ID!, $text: String, $attachStructures: [StructureInput!]) {\n  finishMessage(\n    input: {message: $message, text: $text, attachStructures: $attachStructures}\n  ) {\n    ...ListMessage\n    __typename\n  }\n}'

class GetChromaCollectionQuery(BaseModel):
    """No documentation found for this operation."""
    chroma_collection: ChromaCollection = Field(alias='chromaCollection')
    'Get a single Chroma collection by ID'

    class Arguments(BaseModel):
        """Arguments for GetChromaCollection """
        id: ID

    class Meta:
        """Meta class for GetChromaCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nquery GetChromaCollection($id: ID!) {\n  chromaCollection(id: $id) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class SearchChromaCollectionQueryOptions(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    value: ID
    label: str
    'The human-readable name of the collection, unique within its organization'
    model_config = ConfigDict(frozen=True)

class SearchChromaCollectionQuery(BaseModel):
    """No documentation found for this operation."""
    options: tuple[SearchChromaCollectionQueryOptions, ...]
    "List this organization's Chroma collections"

    class Arguments(BaseModel):
        """Arguments for SearchChromaCollection """
        search: str | None = Field(default=None)
        values: list[ID] | None = Field(default=None)
        limit: Annotated[int | None, GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[int | None, GraphQLDefault('0')] = Field(default=None)

    class Meta:
        """Meta class for SearchChromaCollection """
        document = 'query SearchChromaCollection($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: chromaCollections(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: name\n    __typename\n  }\n}'

class ListChromaCollectionsQuery(BaseModel):
    """No documentation found for this operation."""
    chroma_collections: tuple[ChromaCollection, ...] = Field(alias='chromaCollections')
    "List this organization's Chroma collections"

    class Arguments(BaseModel):
        """Arguments for ListChromaCollections """
        filter: ChromaCollectionFilter | None = Field(default=None)
        order: list[ChromaCollectionOrder] | None = Field(default=None)
        pagination: OffsetPaginationInput | None = Field(default=None)

    class Meta:
        """Meta class for ListChromaCollections """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nquery ListChromaCollections($filter: ChromaCollectionFilter, $order: [ChromaCollectionOrder!], $pagination: OffsetPaginationInput) {\n  chromaCollections(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class QueryDocumentsQuery(BaseModel):
    """No documentation found for this operation."""
    documents: tuple[Document, ...]
    'Search a collection for the documents most similar to some text'

    class Arguments(BaseModel):
        """Arguments for QueryDocuments """
        input: QueryInput

    class Meta:
        """Meta class for QueryDocuments """
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}\n\nquery QueryDocuments($input: QueryInput!) {\n  documents(input: $input) {\n    ...Document\n    __typename\n  }\n}'

class GetLLMModelQuery(BaseModel):
    """No documentation found for this operation."""
    llm_model: LLMModel = Field(alias='llmModel')
    'Get a single LLM model by ID'

    class Arguments(BaseModel):
        """Arguments for GetLLMModel """
        id: ID

    class Meta:
        """Meta class for GetLLMModel """
        document = 'fragment LLMModel on LLMModel {\n  id\n  provider {\n    id\n    name\n    __typename\n  }\n  features\n  embedderFor {\n    id\n    name\n    __typename\n  }\n  __typename\n}\n\nquery GetLLMModel($id: ID!) {\n  llmModel(id: $id) {\n    ...LLMModel\n    __typename\n  }\n}'

class SearchLLMModelsQueryOptions(BaseModel):
    """A LLM model to chage with"""
    typename: Literal['LLMModel'] = Field(alias='__typename', default='LLMModel', exclude=True)
    value: ID
    label: str
    model_config = ConfigDict(frozen=True)

class SearchLLMModelsQuery(BaseModel):
    """No documentation found for this operation."""
    options: tuple[SearchLLMModelsQueryOptions, ...]
    "List the LLM models reachable through this organization's providers"

    class Arguments(BaseModel):
        """Arguments for SearchLLMModels """
        search: str | None = Field(default=None)
        values: list[ID] | None = Field(default=None)
        limit: Annotated[int | None, GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[int | None, GraphQLDefault('0')] = Field(default=None)

    class Meta:
        """Meta class for SearchLLMModels """
        document = 'query SearchLLMModels($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: llmModels(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: modelId\n    __typename\n  }\n}'

class ListLLModelsQuery(BaseModel):
    """No documentation found for this operation."""
    llm_models: tuple[LLMModel, ...] = Field(alias='llmModels')
    "List the LLM models reachable through this organization's providers"

    class Arguments(BaseModel):
        """Arguments for ListLLModels """
        filter: LLMModelFilter | None = Field(default=None)
        order: list[LLMModelOrder] | None = Field(default=None)
        pagination: OffsetPaginationInput | None = Field(default=None)

    class Meta:
        """Meta class for ListLLModels """
        document = 'fragment LLMModel on LLMModel {\n  id\n  provider {\n    id\n    name\n    __typename\n  }\n  features\n  embedderFor {\n    id\n    name\n    __typename\n  }\n  __typename\n}\n\nquery ListLLModels($filter: LLMModelFilter, $order: [LLMModelOrder!], $pagination: OffsetPaginationInput) {\n  llmModels(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...LLMModel\n    __typename\n  }\n}'

class GetMessageQuery(BaseModel):
    """No documentation found for this operation."""
    message: Message
    'Get a single message by ID'

    class Arguments(BaseModel):
        """Arguments for GetMessage """
        id: ID

    class Meta:
        """Meta class for GetMessage """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nquery GetMessage($id: ID!) {\n  message(id: $id) {\n    ...Message\n    __typename\n  }\n}'

class SearchMessagesQueryOptions(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    value: ID
    label: str
    model_config = ConfigDict(frozen=True)

class SearchMessagesQuery(BaseModel):
    """No documentation found for this operation."""
    options: tuple[SearchMessagesQueryOptions, ...]
    "List the messages in this organization's rooms"

    class Arguments(BaseModel):
        """Arguments for SearchMessages """
        search: str | None = Field(default=None)
        values: list[ID] | None = Field(default=None)
        limit: Annotated[int | None, GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[int | None, GraphQLDefault('0')] = Field(default=None)

    class Meta:
        """Meta class for SearchMessages """
        document = 'query SearchMessages($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: messages(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: title\n    __typename\n  }\n}'

class ListMessagesQuery(BaseModel):
    """No documentation found for this operation."""
    messages: tuple[ListMessage, ...]
    "List the messages in this organization's rooms"

    class Arguments(BaseModel):
        """Arguments for ListMessages """
        filter: MessageFilter | None = Field(default=None)
        order: list[MessageOrder] | None = Field(default=None)
        pagination: OffsetPaginationInput | None = Field(default=None)

    class Meta:
        """Meta class for ListMessages """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nquery ListMessages($filter: MessageFilter, $order: [MessageOrder!], $pagination: OffsetPaginationInput) {\n  messages(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...ListMessage\n    __typename\n  }\n}'

class GetRoomQuery(BaseModel):
    """No documentation found for this operation."""
    room: Room
    'Get a single room by ID'

    class Arguments(BaseModel):
        """Arguments for GetRoom """
        id: ID

    class Meta:
        """Meta class for GetRoom """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nquery GetRoom($id: ID!) {\n  room(id: $id) {\n    ...Room\n    __typename\n  }\n}'

class SearchRoomsQueryOptions(BaseModel):
    """A room agents and users converse in"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    value: ID
    label: str
    'The Title of the Room'
    description: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class SearchRoomsQuery(BaseModel):
    """No documentation found for this operation."""
    options: tuple[SearchRoomsQueryOptions, ...]
    'List the rooms in this organization'

    class Arguments(BaseModel):
        """Arguments for SearchRooms """
        search: str | None = Field(default=None)
        values: list[ID] | None = Field(default=None)
        limit: Annotated[int | None, GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[int | None, GraphQLDefault('0')] = Field(default=None)

    class Meta:
        """Meta class for SearchRooms """
        document = 'query SearchRooms($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: rooms(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: title\n    description: description\n    __typename\n  }\n}'

class ListRoomsQuery(BaseModel):
    """No documentation found for this operation."""
    rooms: tuple[Room, ...]
    'List the rooms in this organization'

    class Arguments(BaseModel):
        """Arguments for ListRooms """
        filter: RoomFilter | None = Field(default=None)
        order: list[RoomOrder] | None = Field(default=None)
        pagination: OffsetPaginationInput | None = Field(default=None)

    class Meta:
        """Meta class for ListRooms """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nquery ListRooms($filter: RoomFilter, $order: [RoomOrder!], $pagination: OffsetPaginationInput) {\n  rooms(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...Room\n    __typename\n  }\n}'

class WatchRoomSubscriptionRoomJoin(BaseModel):
    """A participant in a room"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    name: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class WatchRoomSubscriptionRoomLeave(BaseModel):
    """A participant in a room"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    name: str | None = Field(default=None)
    model_config = ConfigDict(frozen=True)

class WatchRoomSubscriptionRoom(BaseModel):
    """Something that happened in a room"""
    typename: Literal['RoomEvent'] = Field(alias='__typename', default='RoomEvent', exclude=True)
    kind: RoomEventKind
    message: ListMessage | None = Field(default=None)
    'The message, for MESSAGE_* events'
    join: WatchRoomSubscriptionRoomJoin | None = Field(default=None)
    'The agent that joined, for JOIN events'
    leave: WatchRoomSubscriptionRoomLeave | None = Field(default=None)
    'The agent that left, for LEAVE events'
    model_config = ConfigDict(frozen=True)

class WatchRoomSubscription(BaseModel):
    """No documentation found for this operation."""
    room: WatchRoomSubscriptionRoom
    'Join a room and receive its events: messages created, streamed into and finished, and agents joining or leaving'

    class Arguments(BaseModel):
        """Arguments for WatchRoom """
        room: ID
        agent_id: ID = Field(validation_alias=AliasChoices('agent_id', 'agentId'), serialization_alias='agentId')
        filter_own: bool | None = Field(validation_alias=AliasChoices('filter_own', 'filterOwn'), serialization_alias='filterOwn', default=None)

    class Meta:
        """Meta class for WatchRoom """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  isStreaming\n  agent {\n    id\n    name\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nsubscription WatchRoom($room: ID!, $agentId: ID!, $filterOwn: Boolean) {\n  room(room: $room, agentId: $agentId, filterOwn: $filterOwn) {\n    kind\n    message {\n      ...ListMessage\n      __typename\n    }\n    join {\n      id\n      name\n      __typename\n    }\n    leave {\n      id\n      name\n      __typename\n    }\n    __typename\n  }\n}'

async def achat(messages: Iterable[ChatMessageInput], model: IDCoercible | None | UnsetType=UNSET, tools: Iterable[ToolInput] | None | UnsetType=UNSET, tool_choice: Any | None | UnsetType=UNSET, temperature: float | None | UnsetType=UNSET, max_tokens: int | None | UnsetType=UNSET, top_p: float | None | UnsetType=UNSET, frequency_penalty: float | None | UnsetType=UNSET, presence_penalty: float | None | UnsetType=UNSET, stop: Iterable[str] | None | UnsetType=UNSET, n: int | None | UnsetType=UNSET, response_format: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChatResponse:
    """Chat 

Send a chat completion request

Args:
    messages: A chat message input (required) (list) (required)
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    tools: A large language model function call (required) (list)
    tool_choice: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    temperature: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    max_tokens: The `Int` scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.
    top_p: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    frequency_penalty: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    presence_penalty: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    stop: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required) (list)
    n: The `Int` scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.
    response_format: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChatResponse
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['messages'] = messages
    if model is not UNSET:
        _input['model'] = model
    if tools is not UNSET:
        _input['tools'] = tools
    if tool_choice is not UNSET:
        _input['toolChoice'] = tool_choice
    if temperature is not UNSET:
        _input['temperature'] = temperature
    if max_tokens is not UNSET:
        _input['maxTokens'] = max_tokens
    if top_p is not UNSET:
        _input['topP'] = top_p
    if frequency_penalty is not UNSET:
        _input['frequencyPenalty'] = frequency_penalty
    if presence_penalty is not UNSET:
        _input['presencePenalty'] = presence_penalty
    if stop is not UNSET:
        _input['stop'] = stop
    if n is not UNSET:
        _input['n'] = n
    if response_format is not UNSET:
        _input['responseFormat'] = response_format
    variables['input'] = _input
    return (await aexecute(ChatMutation, variables, rath=rath)).chat

def chat(messages: Iterable[ChatMessageInput], model: IDCoercible | None | UnsetType=UNSET, tools: Iterable[ToolInput] | None | UnsetType=UNSET, tool_choice: Any | None | UnsetType=UNSET, temperature: float | None | UnsetType=UNSET, max_tokens: int | None | UnsetType=UNSET, top_p: float | None | UnsetType=UNSET, frequency_penalty: float | None | UnsetType=UNSET, presence_penalty: float | None | UnsetType=UNSET, stop: Iterable[str] | None | UnsetType=UNSET, n: int | None | UnsetType=UNSET, response_format: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChatResponse:
    """Chat 

Send a chat completion request

Args:
    messages: A chat message input (required) (list) (required)
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    tools: A large language model function call (required) (list)
    tool_choice: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    temperature: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    max_tokens: The `Int` scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.
    top_p: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    frequency_penalty: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    presence_penalty: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    stop: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required) (list)
    n: The `Int` scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.
    response_format: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChatResponse
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['messages'] = messages
    if model is not UNSET:
        _input['model'] = model
    if tools is not UNSET:
        _input['tools'] = tools
    if tool_choice is not UNSET:
        _input['toolChoice'] = tool_choice
    if temperature is not UNSET:
        _input['temperature'] = temperature
    if max_tokens is not UNSET:
        _input['maxTokens'] = max_tokens
    if top_p is not UNSET:
        _input['topP'] = top_p
    if frequency_penalty is not UNSET:
        _input['frequencyPenalty'] = frequency_penalty
    if presence_penalty is not UNSET:
        _input['presencePenalty'] = presence_penalty
    if stop is not UNSET:
        _input['stop'] = stop
    if n is not UNSET:
        _input['n'] = n
    if response_format is not UNSET:
        _input['responseFormat'] = response_format
    variables['input'] = _input
    return execute(ChatMutation, variables, rath=rath).chat

async def acreate_collection(name: str, embedder: IDCoercible, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChromaCollection:
    """CreateCollection 

Create a searchable collection of documents

Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    variables['input'] = _input
    return (await aexecute(CreateCollectionMutation, variables, rath=rath)).create_collection

def create_collection(name: str, embedder: IDCoercible, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChromaCollection:
    """CreateCollection 

Create a searchable collection of documents

Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    variables['input'] = _input
    return execute(CreateCollectionMutation, variables, rath=rath).create_collection

async def aensure_collection(name: str, embedder: IDCoercible, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChromaCollection:
    """EnsureCollection 

Create a collection, or update it if it already exists

Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    variables['input'] = _input
    return (await aexecute(EnsureCollectionMutation, variables, rath=rath)).ensure_collection

def ensure_collection(name: str, embedder: IDCoercible, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ChromaCollection:
    """EnsureCollection 

Create a collection, or update it if it already exists

Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    variables['input'] = _input
    return execute(EnsureCollectionMutation, variables, rath=rath).ensure_collection

async def aadd_documents_to_collection(collection: IDCoercible, documents: Iterable[DocumentInput], rath: AlpakaRath | None=None) -> tuple[Document, ...]:
    """AddDocumentsToCollection 

Embed documents and add them to a collection

Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    documents: A document to put into the vector database (required) (list) (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Document]
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['collection'] = collection
    _input['documents'] = documents
    variables['input'] = _input
    return (await aexecute(AddDocumentsToCollectionMutation, variables, rath=rath)).add_documents_to_collection

def add_documents_to_collection(collection: IDCoercible, documents: Iterable[DocumentInput], rath: AlpakaRath | None=None) -> tuple[Document, ...]:
    """AddDocumentsToCollection 

Embed documents and add them to a collection

Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    documents: A document to put into the vector database (required) (list) (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Document]
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['collection'] = collection
    _input['documents'] = documents
    variables['input'] = _input
    return execute(AddDocumentsToCollectionMutation, variables, rath=rath).add_documents_to_collection

async def agenerate_image(description: str, model: IDCoercible | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> GenerateImageMutationGenerateImage:
    """GenerateImage 

Generate an image from a text description

Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    GenerateImageMutationGenerateImage
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['description'] = description
    variables['input'] = _input
    return (await aexecute(GenerateImageMutation, variables, rath=rath)).generate_image

def generate_image(description: str, model: IDCoercible | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> GenerateImageMutationGenerateImage:
    """GenerateImage 

Generate an image from a text description

Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    GenerateImageMutationGenerateImage
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['description'] = description
    variables['input'] = _input
    return execute(GenerateImageMutation, variables, rath=rath).generate_image

async def asend(text: str, room: IDCoercible, agent_id: str, attach_structures: list[StructureInput] | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Message:
    """Send 

Post a complete message into a room

Args:
    text (str): No description
    room (ID): No description
    agent_id (str): No description
    attach_structures (list[StructureInput] | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: dict[str, Any] = {}
    variables['text'] = text
    variables['room'] = room
    variables['agentId'] = agent_id
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return (await aexecute(SendMutation, variables, rath=rath)).send

def send(text: str, room: IDCoercible, agent_id: str, attach_structures: list[StructureInput] | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Message:
    """Send 

Post a complete message into a room

Args:
    text (str): No description
    room (ID): No description
    agent_id (str): No description
    attach_structures (list[StructureInput] | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: dict[str, Any] = {}
    variables['text'] = text
    variables['room'] = room
    variables['agentId'] = agent_id
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return execute(SendMutation, variables, rath=rath).send

async def acreate_provider(kind: ProviderKind, description: str | None | UnsetType=UNSET, name: str | None | UnsetType=UNSET, api_key: str | None | UnsetType=UNSET, api_base: str | None | UnsetType=UNSET, additional_config: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Provider:
    """CreateProvider 

Configure a new LLM provider and list the models it offers

Args:
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    kind: ProviderKind (required)
    api_key: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    api_base: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    additional_config: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Provider
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    if description is not UNSET:
        _input['description'] = description
    if name is not UNSET:
        _input['name'] = name
    _input['kind'] = kind
    if api_key is not UNSET:
        _input['apiKey'] = api_key
    if api_base is not UNSET:
        _input['apiBase'] = api_base
    if additional_config is not UNSET:
        _input['additionalConfig'] = additional_config
    variables['input'] = _input
    return (await aexecute(CreateProviderMutation, variables, rath=rath)).create_provider

def create_provider(kind: ProviderKind, description: str | None | UnsetType=UNSET, name: str | None | UnsetType=UNSET, api_key: str | None | UnsetType=UNSET, api_base: str | None | UnsetType=UNSET, additional_config: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Provider:
    """CreateProvider 

Configure a new LLM provider and list the models it offers

Args:
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    kind: ProviderKind (required)
    api_key: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    api_base: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    additional_config: The `JSON` scalar type represents JSON values as specified by [ECMA-404](https://ecma-international.org/wp-content/uploads/ECMA-404_2nd_edition_december_2017.pdf).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Provider
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    if description is not UNSET:
        _input['description'] = description
    if name is not UNSET:
        _input['name'] = name
    _input['kind'] = kind
    if api_key is not UNSET:
        _input['apiKey'] = api_key
    if api_base is not UNSET:
        _input['apiBase'] = api_base
    if additional_config is not UNSET:
        _input['additionalConfig'] = additional_config
    variables['input'] = _input
    return execute(CreateProviderMutation, variables, rath=rath).create_provider

async def apull(model_name: str, provider: IDCoercible | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> PullMutationPull:
    """Pull 

Pull a model into an Ollama provider

Args:
    model_name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    provider: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    PullMutationPull
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['modelName'] = model_name
    if provider is not UNSET:
        _input['provider'] = provider
    variables['input'] = _input
    return (await aexecute(PullMutation, variables, rath=rath)).pull

def pull(model_name: str, provider: IDCoercible | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> PullMutationPull:
    """Pull 

Pull a model into an Ollama provider

Args:
    model_name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    provider: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    PullMutationPull
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['modelName'] = model_name
    if provider is not UNSET:
        _input['provider'] = provider
    variables['input'] = _input
    return execute(PullMutation, variables, rath=rath).pull

async def acreate_room(title: str | None | UnsetType=UNSET, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Room:
    """CreateRoom 

Open a new room

Args:
    title (str | None, optional): No description. 
    description (str | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: dict[str, Any] = {}
    if title is not UNSET:
        variables['title'] = title
    if description is not UNSET:
        variables['description'] = description
    return (await aexecute(CreateRoomMutation, variables, rath=rath)).create_room

def create_room(title: str | None | UnsetType=UNSET, description: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Room:
    """CreateRoom 

Open a new room

Args:
    title (str | None, optional): No description. 
    description (str | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: dict[str, Any] = {}
    if title is not UNSET:
        variables['title'] = title
    if description is not UNSET:
        variables['description'] = description
    return execute(CreateRoomMutation, variables, rath=rath).create_room

async def astart_message(room: IDCoercible, agent_id: str, parent: IDCoercible | None | UnsetType=UNSET, text: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ListMessage:
    """StartMessage 

Open a message to stream text into. Only the starting agent (same user and client) can append to or finish it.

Args:
    room (ID): No description
    agent_id (str): No description
    parent (ID | None, optional): No description. 
    text (str | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
    if parent is not UNSET:
        variables['parent'] = parent
    if text is not UNSET:
        variables['text'] = text
    return (await aexecute(StartMessageMutation, variables, rath=rath)).start_message

def start_message(room: IDCoercible, agent_id: str, parent: IDCoercible | None | UnsetType=UNSET, text: str | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ListMessage:
    """StartMessage 

Open a message to stream text into. Only the starting agent (same user and client) can append to or finish it.

Args:
    room (ID): No description
    agent_id (str): No description
    parent (ID | None, optional): No description. 
    text (str | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
    if parent is not UNSET:
        variables['parent'] = parent
    if text is not UNSET:
        variables['text'] = text
    return execute(StartMessageMutation, variables, rath=rath).start_message

async def aappend_message(message: IDCoercible, delta: str, rath: AlpakaRath | None=None) -> ListMessage:
    """AppendMessage 

Append a delta to a streaming message. Batch deltas (every ~100-250 ms or ~30 characters) and await each call before sending the next, so they arrive in order.

Args:
    message (ID): No description
    delta (str): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['message'] = message
    variables['delta'] = delta
    return (await aexecute(AppendMessageMutation, variables, rath=rath)).append_message

def append_message(message: IDCoercible, delta: str, rath: AlpakaRath | None=None) -> ListMessage:
    """AppendMessage 

Append a delta to a streaming message. Batch deltas (every ~100-250 ms or ~30 characters) and await each call before sending the next, so they arrive in order.

Args:
    message (ID): No description
    delta (str): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['message'] = message
    variables['delta'] = delta
    return execute(AppendMessageMutation, variables, rath=rath).append_message

async def afinish_message(message: IDCoercible, text: str | None | UnsetType=UNSET, attach_structures: list[StructureInput] | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ListMessage:
    """FinishMessage 

Close a streaming message. Pass the full final text so any delta lost on the way is repaired; call it in a finally block so a crashed stream never stays open.

Args:
    message (ID): No description
    text (str | None, optional): No description. 
    attach_structures (list[StructureInput] | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['message'] = message
    if text is not UNSET:
        variables['text'] = text
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return (await aexecute(FinishMessageMutation, variables, rath=rath)).finish_message

def finish_message(message: IDCoercible, text: str | None | UnsetType=UNSET, attach_structures: list[StructureInput] | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> ListMessage:
    """FinishMessage 

Close a streaming message. Pass the full final text so any delta lost on the way is repaired; call it in a finally block so a crashed stream never stays open.

Args:
    message (ID): No description
    text (str | None, optional): No description. 
    attach_structures (list[StructureInput] | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ListMessage
"""
    variables: dict[str, Any] = {}
    variables['message'] = message
    if text is not UNSET:
        variables['text'] = text
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return execute(FinishMessageMutation, variables, rath=rath).finish_message

async def aget_chroma_collection(id: IDCoercible, rath: AlpakaRath | None=None) -> ChromaCollection:
    """GetChromaCollection 

Get a single Chroma collection by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetChromaCollectionQuery, variables, rath=rath)).chroma_collection

def get_chroma_collection(id: IDCoercible, rath: AlpakaRath | None=None) -> ChromaCollection:
    """GetChromaCollection 

Get a single Chroma collection by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return execute(GetChromaCollectionQuery, variables, rath=rath).chroma_collection

async def asearch_chroma_collection(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchChromaCollectionQueryOptions, ...]:
    """SearchChromaCollection 

List this organization's Chroma collections

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchChromaCollectionQueryChromaCollections]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchChromaCollectionQuery, variables, rath=rath)).options

def search_chroma_collection(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchChromaCollectionQueryOptions, ...]:
    """SearchChromaCollection 

List this organization's Chroma collections

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchChromaCollectionQueryChromaCollections]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchChromaCollectionQuery, variables, rath=rath).options

async def alist_chroma_collections(filter: ChromaCollectionFilter | None | UnsetType=UNSET, order: list[ChromaCollectionOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[ChromaCollection, ...]:
    """ListChromaCollections 

List this organization's Chroma collections

Args:
    filter (ChromaCollectionFilter | None, optional): No description. 
    order (list[ChromaCollectionOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[ChromaCollection]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListChromaCollectionsQuery, variables, rath=rath)).chroma_collections

def list_chroma_collections(filter: ChromaCollectionFilter | None | UnsetType=UNSET, order: list[ChromaCollectionOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[ChromaCollection, ...]:
    """ListChromaCollections 

List this organization's Chroma collections

Args:
    filter (ChromaCollectionFilter | None, optional): No description. 
    order (list[ChromaCollectionOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[ChromaCollection]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListChromaCollectionsQuery, variables, rath=rath).chroma_collections

async def aquery_documents(collection: IDCoercible, query_texts: Iterable[str], n_results: int, where: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[Document, ...]:
    """QueryDocuments 

Search a collection for the documents most similar to some text

Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    query_texts: One or more query texts; the union of their results is returned, deduplicated by document
    n_results: Results per query text
    where: Chroma metadata filter applied to every query
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Document]
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['collection'] = collection
    _input['queryTexts'] = query_texts
    _input['nResults'] = n_results
    if where is not UNSET:
        _input['where'] = where
    variables['input'] = _input
    return (await aexecute(QueryDocumentsQuery, variables, rath=rath)).documents

def query_documents(collection: IDCoercible, query_texts: Iterable[str], n_results: int, where: Any | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[Document, ...]:
    """QueryDocuments 

Search a collection for the documents most similar to some text

Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    query_texts: One or more query texts; the union of their results is returned, deduplicated by document
    n_results: Results per query text
    where: Chroma metadata filter applied to every query
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Document]
"""
    variables: dict[str, Any] = {}
    _input: dict[str, Any] = {}
    _input['collection'] = collection
    _input['queryTexts'] = query_texts
    _input['nResults'] = n_results
    if where is not UNSET:
        _input['where'] = where
    variables['input'] = _input
    return execute(QueryDocumentsQuery, variables, rath=rath).documents

async def aget_llm_model(id: IDCoercible, rath: AlpakaRath | None=None) -> LLMModel:
    """GetLLMModel 

Get a single LLM model by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    LLMModel
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetLLMModelQuery, variables, rath=rath)).llm_model

def get_llm_model(id: IDCoercible, rath: AlpakaRath | None=None) -> LLMModel:
    """GetLLMModel 

Get a single LLM model by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    LLMModel
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return execute(GetLLMModelQuery, variables, rath=rath).llm_model

async def asearch_llm_models(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchLLMModelsQueryOptions, ...]:
    """SearchLLMModels 

List the LLM models reachable through this organization's providers

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchLLMModelsQueryLlmModels]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchLLMModelsQuery, variables, rath=rath)).options

def search_llm_models(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchLLMModelsQueryOptions, ...]:
    """SearchLLMModels 

List the LLM models reachable through this organization's providers

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchLLMModelsQueryLlmModels]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchLLMModelsQuery, variables, rath=rath).options

async def alist_ll_models(filter: LLMModelFilter | None | UnsetType=UNSET, order: list[LLMModelOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[LLMModel, ...]:
    """ListLLModels 

List the LLM models reachable through this organization's providers

Args:
    filter (LLMModelFilter | None, optional): No description. 
    order (list[LLMModelOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[LLMModel]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListLLModelsQuery, variables, rath=rath)).llm_models

def list_ll_models(filter: LLMModelFilter | None | UnsetType=UNSET, order: list[LLMModelOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[LLMModel, ...]:
    """ListLLModels 

List the LLM models reachable through this organization's providers

Args:
    filter (LLMModelFilter | None, optional): No description. 
    order (list[LLMModelOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[LLMModel]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListLLModelsQuery, variables, rath=rath).llm_models

async def aget_message(id: IDCoercible, rath: AlpakaRath | None=None) -> Message:
    """GetMessage 

Get a single message by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetMessageQuery, variables, rath=rath)).message

def get_message(id: IDCoercible, rath: AlpakaRath | None=None) -> Message:
    """GetMessage 

Get a single message by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return execute(GetMessageQuery, variables, rath=rath).message

async def asearch_messages(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchMessagesQueryOptions, ...]:
    """SearchMessages 

List the messages in this organization's rooms

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchMessagesQueryMessages]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchMessagesQuery, variables, rath=rath)).options

def search_messages(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchMessagesQueryOptions, ...]:
    """SearchMessages 

List the messages in this organization's rooms

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchMessagesQueryMessages]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchMessagesQuery, variables, rath=rath).options

async def alist_messages(filter: MessageFilter | None | UnsetType=UNSET, order: list[MessageOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[ListMessage, ...]:
    """ListMessages 

List the messages in this organization's rooms

Args:
    filter (MessageFilter | None, optional): No description. 
    order (list[MessageOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[ListMessage]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListMessagesQuery, variables, rath=rath)).messages

def list_messages(filter: MessageFilter | None | UnsetType=UNSET, order: list[MessageOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[ListMessage, ...]:
    """ListMessages 

List the messages in this organization's rooms

Args:
    filter (MessageFilter | None, optional): No description. 
    order (list[MessageOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[ListMessage]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListMessagesQuery, variables, rath=rath).messages

async def aget_room(id: IDCoercible, rath: AlpakaRath | None=None) -> Room:
    """GetRoom 

Get a single room by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetRoomQuery, variables, rath=rath)).room

def get_room(id: IDCoercible, rath: AlpakaRath | None=None) -> Room:
    """GetRoom 

Get a single room by ID

Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: dict[str, Any] = {}
    variables['id'] = id
    return execute(GetRoomQuery, variables, rath=rath).room

async def asearch_rooms(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchRoomsQueryOptions, ...]:
    """SearchRooms 

List the rooms in this organization

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchRoomsQueryRooms]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchRoomsQuery, variables, rath=rath)).options

def search_rooms(search: str | None | UnsetType=UNSET, values: list[IDCoercible] | None | UnsetType=UNSET, limit: int | None | UnsetType=UNSET, offset: int | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[SearchRoomsQueryOptions, ...]:
    """SearchRooms 

List the rooms in this organization

Args:
    search (str | None, optional): No description. 
    values (list[ID] | None, optional): No description. 
    limit (int | None, optional): No description. Defaults to 10
    offset (int | None, optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[SearchRoomsQueryRooms]
"""
    variables: dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchRoomsQuery, variables, rath=rath).options

async def alist_rooms(filter: RoomFilter | None | UnsetType=UNSET, order: list[RoomOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[Room, ...]:
    """ListRooms 

List the rooms in this organization

Args:
    filter (RoomFilter | None, optional): No description. 
    order (list[RoomOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Room]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListRoomsQuery, variables, rath=rath)).rooms

def list_rooms(filter: RoomFilter | None | UnsetType=UNSET, order: list[RoomOrder] | None | UnsetType=UNSET, pagination: OffsetPaginationInput | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> tuple[Room, ...]:
    """ListRooms 

List the rooms in this organization

Args:
    filter (RoomFilter | None, optional): No description. 
    order (list[RoomOrder] | None, optional): No description. 
    pagination (OffsetPaginationInput | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    list[Room]
"""
    variables: dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListRoomsQuery, variables, rath=rath).rooms

async def awatch_room(room: IDCoercible, agent_id: IDCoercible, filter_own: bool | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> AsyncIterator[WatchRoomSubscriptionRoom]:
    """WatchRoom 

Join a room and receive its events: messages created, streamed into and finished, and agents joining or leaving

Args:
    room (ID): No description
    agent_id (ID): No description
    filter_own (bool | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    WatchRoomSubscriptionRoom
"""
    variables: dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
    if filter_own is not UNSET:
        variables['filterOwn'] = filter_own
    async for event in asubscribe(WatchRoomSubscription, variables, rath=rath):
        yield event.room

def watch_room(room: IDCoercible, agent_id: IDCoercible, filter_own: bool | None | UnsetType=UNSET, rath: AlpakaRath | None=None) -> Iterator[WatchRoomSubscriptionRoom]:
    """WatchRoom 

Join a room and receive its events: messages created, streamed into and finished, and agents joining or leaving

Args:
    room (ID): No description
    agent_id (ID): No description
    filter_own (bool | None, optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    WatchRoomSubscriptionRoom
"""
    variables: dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
    if filter_own is not UNSET:
        variables['filterOwn'] = filter_own
    for event in subscribe(WatchRoomSubscription, variables, rath=rath):
        yield event.room
AddDocumentsToCollectionInput.model_rebuild()
ChatInput.model_rebuild()
ChatMessageInput.model_rebuild()
ChromaCollectionFilter.model_rebuild()
DocumentInput.model_rebuild()
LLMModelFilter.model_rebuild()
MessageFilter.model_rebuild()
RoomFilter.model_rebuild()