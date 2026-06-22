from rath.scalars import ID, IDCoercible
from pydantic import ConfigDict, BaseModel, Field
from typing import Any, AsyncIterator, Dict, Iterator, Annotated, Literal, Tuple, Iterable, Optional, Union, List
from alpaka.funcs import aexecute, asubscribe, subscribe, execute
from alpaka.rath import AlpakaRath
from datetime import datetime
from alpaka.traits import ChatResponseTraits
from enum import Enum

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

class GraphQLDefault:
    """Records a GraphQL field schema default value. The client omits the field so the server applies its own default; this preserves the value for introspection."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'GraphQLDefault(' + repr(self.value) + ')'

class FeatureType(str, Enum):
    """The type of the thinking block"""
    EMBEDDING = 'EMBEDDING'
    CHATTING = 'CHATTING'
    CHAT = 'CHAT'

class InputModality(str, Enum):
    """Modalities"""
    IMAGE = 'IMAGE'
    TEXT = 'TEXT'
    AUDIO = 'AUDIO'
    VIDEO = 'VIDEO'

class Ordering(str, Enum):
    """No documentation"""
    ASC = 'ASC'
    ASC_NULLS_FIRST = 'ASC_NULLS_FIRST'
    ASC_NULLS_LAST = 'ASC_NULLS_LAST'
    DESC = 'DESC'
    DESC_NULLS_FIRST = 'DESC_NULLS_FIRST'
    DESC_NULLS_LAST = 'DESC_NULLS_LAST'

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

class Role(str, Enum):
    """The type of the message sender"""
    SYSTEM = 'SYSTEM'
    USER = 'USER'
    ASSISTANT = 'ASSISTANT'
    TOOL = 'TOOL'
    FUNCTION = 'FUNCTION'

class ToolType(str, Enum):
    """The type of the tool"""
    FUNCTION = 'FUNCTION'

class AddDocumentsToCollectionInput(BaseModel):
    """No documentation"""
    collection: ID
    documents: Tuple['DocumentInput', ...]
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChatInput(BaseModel):
    """A chat message input"""
    model: Optional[ID] = None
    messages: Tuple['ChatMessageInput', ...]
    tools: Optional[Tuple['ToolInput', ...]] = None
    temperature: Optional[float] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChatMessageInput(BaseModel):
    """A chat message input"""
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = Field(alias='toolCallId', default=None)
    function_call: Optional['FunctionCallInput'] = Field(alias='functionCall', default=None)
    tool_calls: Optional[Tuple['ToolCallInput', ...]] = Field(alias='toolCalls', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionFilter(BaseModel):
    """Filter for ChromaCollection"""
    and_: Optional['ChromaCollectionFilter'] = Field(alias='AND', default=None)
    or_: Optional['ChromaCollectionFilter'] = Field(alias='OR', default=None)
    not_: Optional['ChromaCollectionFilter'] = Field(alias='NOT', default=None)
    distinct: Optional[bool] = Field(alias='DISTINCT', default=None)
    ids: Optional[Tuple[ID, ...]] = None
    search: Optional[str] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionInput(BaseModel):
    """No documentation"""
    name: str
    embedder: ID
    description: Optional[str] = None
    is_public: Annotated[Optional[bool], GraphQLDefault('False')] = Field(alias='isPublic', default=None)
    'Default: False'
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollectionOrder(BaseModel):
    """No documentation"""
    name: Optional[Ordering] = None
    created_at: Optional[Ordering] = Field(alias='createdAt', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class DocumentInput(BaseModel):
    """A document to put into the vector database"""
    content: str
    structure: Optional['StructureInput'] = None
    id: Optional[str] = None
    metadata: Optional[Any] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class FunctionCallInput(BaseModel):
    """A function call input"""
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class FunctionDefinitionInput(BaseModel):
    """A large language model function defintion"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Any] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ImageInput(BaseModel):
    """The image"""
    model: Optional[ID] = None
    description: str
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class LLMModelFilter(BaseModel):
    """Filter for LLMModel"""
    and_: Optional['LLMModelFilter'] = Field(alias='AND', default=None)
    or_: Optional['LLMModelFilter'] = Field(alias='OR', default=None)
    not_: Optional['LLMModelFilter'] = Field(alias='NOT', default=None)
    distinct: Optional[bool] = Field(alias='DISTINCT', default=None)
    ids: Optional[Tuple[ID, ...]] = None
    search: Optional[str] = None
    input_modalities: Optional[Tuple[InputModality, ...]] = Field(alias='inputModalities', default=None)
    output_modalities: Optional[Tuple[InputModality, ...]] = Field(alias='outputModalities', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class LLMModelOrder(BaseModel):
    """No documentation"""
    label: Optional[Ordering] = None
    model_id: Optional[Ordering] = Field(alias='modelId', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class MessageFilter(BaseModel):
    """Message represent the message of an agent on a room"""
    and_: Optional['MessageFilter'] = Field(alias='AND', default=None)
    or_: Optional['MessageFilter'] = Field(alias='OR', default=None)
    not_: Optional['MessageFilter'] = Field(alias='NOT', default=None)
    distinct: Optional[bool] = Field(alias='DISTINCT', default=None)
    ids: Optional[Tuple[ID, ...]] = None
    search: Optional[str] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class MessageOrder(BaseModel):
    """No documentation"""
    created_at: Optional[Ordering] = Field(alias='createdAt', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class OffsetPaginationInput(BaseModel):
    """No documentation"""
    offset: Annotated[Optional[int], GraphQLDefault('0')] = None
    'Default: 0'
    limit: Optional[int] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ProviderInput(BaseModel):
    """A large language model to change with"""
    description: Optional[str] = None
    name: Optional[str] = None
    kind: ProviderKind
    api_key: Optional[str] = Field(alias='apiKey', default=None)
    api_base: Optional[str] = Field(alias='apiBase', default=None)
    additional_config: Optional[Any] = Field(alias='additionalConfig', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class PullInput(BaseModel):
    """No documentation"""
    model_name: str = Field(alias='modelName')
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class RoomFilter(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    and_: Optional['RoomFilter'] = Field(alias='AND', default=None)
    or_: Optional['RoomFilter'] = Field(alias='OR', default=None)
    not_: Optional['RoomFilter'] = Field(alias='NOT', default=None)
    distinct: Optional[bool] = Field(alias='DISTINCT', default=None)
    ids: Optional[Tuple[ID, ...]] = None
    search: Optional[str] = None
    talking_about: Optional['StructureInput'] = Field(alias='talkingAbout', default=None)
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class RoomOrder(BaseModel):
    """No documentation"""
    created_at: Optional[Ordering] = Field(alias='createdAt', default=None)
    title: Optional[Ordering] = None
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class StructureInput(BaseModel):
    """A function definition for a large language model"""
    identifier: str
    object: str
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ToolCallInput(BaseModel):
    """A tool call input"""
    id: str
    function: FunctionCallInput
    type: ToolType
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ToolInput(BaseModel):
    """A large language model function call"""
    type: Annotated[Optional[ToolType], GraphQLDefault('FUNCTION')] = None
    'Default: FUNCTION'
    function: FunctionDefinitionInput
    model_config = ConfigDict(frozen=True, extra='forbid', populate_by_name=True, use_enum_values=True)

class ChromaCollection(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    id: ID
    name: str
    description: str
    created_at: datetime = Field(alias='createdAt')
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ChromaCollection"""
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}'
        name = 'ChromaCollection'
        type = 'ChromaCollection'

class Document(BaseModel):
    """No documentation"""
    typename: Literal['Document'] = Field(alias='__typename', default='Document', exclude=True)
    id: str
    content: str
    metadata: Optional[Any] = Field(default=None)
    distance: Optional[float] = Field(default=None)
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Document"""
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}'
        name = 'Document'
        type = 'Document'

class LLMModelProvider(BaseModel):
    """A provider of LLMs"""
    typename: Literal['Provider'] = Field(alias='__typename', default='Provider', exclude=True)
    id: str
    name: str
    model_config = ConfigDict(frozen=True)

class LLMModelEmbedderfor(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    id: ID
    name: str
    model_config = ConfigDict(frozen=True)

class LLMModel(BaseModel):
    """A LLM model to chage with"""
    typename: Literal['LLMModel'] = Field(alias='__typename', default='LLMModel', exclude=True)
    id: ID
    provider: LLMModelProvider
    features: Tuple[FeatureType, ...]
    'The features supported by the model'
    embedder_for: Tuple[LLMModelEmbedderfor, ...] = Field(alias='embedderFor')
    'The collections that can be embedded with this model'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for LLMModel"""
        document = 'fragment LLMModel on LLMModel {\n  id\n  provider {\n    id\n    name\n    __typename\n  }\n  features\n  embedderFor {\n    id\n    name\n    __typename\n  }\n  __typename\n}'
        name = 'LLMModel'
        type = 'LLMModel'

class ListMessageAgent(BaseModel):
    """Agent(id, room, name, client, user)"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    model_config = ConfigDict(frozen=True)

class ListMessageAttachedstructures(BaseModel):
    """The type of the tool"""
    typename: Literal['Structure'] = Field(alias='__typename', default='Structure', exclude=True)
    object: str
    identifier: str
    model_config = ConfigDict(frozen=True)

class ListMessage(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    id: ID
    text: str
    'A clear text representation of the rich comment'
    agent: ListMessageAgent
    'The user that created this comment'
    attached_structures: Tuple[ListMessageAttachedstructures, ...] = Field(alias='attachedStructures')
    'The collections that can be embedded with this model'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ListMessage"""
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}'
        name = 'ListMessage'
        type = 'Message'

class ProviderModels(BaseModel):
    """A LLM model to chage with"""
    typename: Literal['LLMModel'] = Field(alias='__typename', default='LLMModel', exclude=True)
    id: ID
    model_id: str = Field(alias='modelId')
    features: Tuple[FeatureType, ...]
    'The features supported by the model'
    model_config = ConfigDict(frozen=True)

class Provider(BaseModel):
    """A provider of LLMs"""
    typename: Literal['Provider'] = Field(alias='__typename', default='Provider', exclude=True)
    id: str
    name: str
    models: Tuple[ProviderModels, ...]
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

class ChatResponseChoicesMessageFunctioncall(BaseModel):
    """The type of the tool"""
    typename: Literal['FunctionCall'] = Field(alias='__typename', default='FunctionCall', exclude=True)
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessageToolcallsFunction(BaseModel):
    """The type of the tool"""
    typename: Literal['FunctionCall'] = Field(alias='__typename', default='FunctionCall', exclude=True)
    name: str
    arguments: str
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessageToolcalls(BaseModel):
    """A function definition for a large language model"""
    typename: Literal['ToolCall'] = Field(alias='__typename', default='ToolCall', exclude=True)
    id: str
    type: ToolType
    function: ChatResponseChoicesMessageToolcallsFunction
    model_config = ConfigDict(frozen=True)

class ChatResponseChoicesMessage(BaseModel):
    """No documentation"""
    typename: Literal['ChatMessage'] = Field(alias='__typename', default='ChatMessage', exclude=True)
    role: Role
    content: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None, alias='toolCallId')
    function_call: Optional[ChatResponseChoicesMessageFunctioncall] = Field(default=None, alias='functionCall')
    tool_calls: Optional[Tuple[ChatResponseChoicesMessageToolcalls, ...]] = Field(default=None, alias='toolCalls')
    model_config = ConfigDict(frozen=True)

class ChatResponseChoices(BaseModel):
    """No documentation"""
    typename: Literal['Choice'] = Field(alias='__typename', default='Choice', exclude=True)
    index: int
    finish_reason: Optional[str] = Field(default=None, alias='finishReason')
    message: ChatResponseChoicesMessage
    model_config = ConfigDict(frozen=True)

class ChatResponse(ChatResponseTraits, BaseModel):
    """No documentation"""
    typename: Literal['ChatResponse'] = Field(alias='__typename', default='ChatResponse', exclude=True)
    id: str
    object: str
    created: int
    model: str
    usage: Optional[ChatResponseUsage] = Field(default=None)
    choices: Tuple[ChatResponseChoices, ...]
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for ChatResponse"""
        document = 'fragment ChatResponse on ChatResponse {\n  id\n  object\n  created\n  model\n  usage {\n    promptTokens\n    completionTokens\n    totalTokens\n    __typename\n  }\n  choices {\n    index\n    finishReason\n    message {\n      role\n      content\n      name\n      toolCallId\n      functionCall {\n        name\n        arguments\n        __typename\n      }\n      toolCalls {\n        id\n        type\n        function {\n          name\n          arguments\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  __typename\n}'
        name = 'ChatResponse'
        type = 'ChatResponse'

class Room(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    title: str
    'The Title of the Room'
    description: str
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Room"""
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}'
        name = 'Room'
        type = 'Room'

class MessageAgentRoom(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    model_config = ConfigDict(frozen=True)

class MessageAgent(BaseModel):
    """Agent(id, room, name, client, user)"""
    typename: Literal['Agent'] = Field(alias='__typename', default='Agent', exclude=True)
    id: ID
    room: MessageAgentRoom
    model_config = ConfigDict(frozen=True)

class MessageRoom(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    id: ID
    title: str
    'The Title of the Room'
    model_config = ConfigDict(frozen=True)

class MessageAttachedstructures(BaseModel):
    """The type of the tool"""
    typename: Literal['Structure'] = Field(alias='__typename', default='Structure', exclude=True)
    object: str
    identifier: str
    model_config = ConfigDict(frozen=True)

class Message(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    id: ID
    text: str
    'A clear text representation of the rich comment'
    agent: MessageAgent
    'The user that created this comment'
    room: MessageRoom
    before: Tuple[ListMessage, ...]
    attached_structures: Tuple[MessageAttachedstructures, ...] = Field(alias='attachedStructures')
    'The collections that can be embedded with this model'
    model_config = ConfigDict(frozen=True)

    class Meta:
        """Meta class for Message"""
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  agent {\n    id\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}'
        name = 'Message'
        type = 'Message'

class ChatMutation(BaseModel):
    """No documentation found for this operation."""
    chat: ChatResponse

    class Arguments(BaseModel):
        """Arguments for Chat """
        input: ChatInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for Chat """
        document = 'fragment ChatResponse on ChatResponse {\n  id\n  object\n  created\n  model\n  usage {\n    promptTokens\n    completionTokens\n    totalTokens\n    __typename\n  }\n  choices {\n    index\n    finishReason\n    message {\n      role\n      content\n      name\n      toolCallId\n      functionCall {\n        name\n        arguments\n        __typename\n      }\n      toolCalls {\n        id\n        type\n        function {\n          name\n          arguments\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nmutation Chat($input: ChatInput!) {\n  chat(input: $input) {\n    ...ChatResponse\n    __typename\n  }\n}'

class CreateCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    create_collection: ChromaCollection = Field(alias='createCollection')

    class Arguments(BaseModel):
        """Arguments for CreateCollection """
        input: ChromaCollectionInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for CreateCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nmutation CreateCollection($input: ChromaCollectionInput!) {\n  createCollection(input: $input) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class EnsureCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    ensure_collection: ChromaCollection = Field(alias='ensureCollection')

    class Arguments(BaseModel):
        """Arguments for EnsureCollection """
        input: ChromaCollectionInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for EnsureCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nmutation EnsureCollection($input: ChromaCollectionInput!) {\n  ensureCollection(input: $input) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class AddDocumentsToCollectionMutation(BaseModel):
    """No documentation found for this operation."""
    add_documents_to_collection: Tuple[Document, ...] = Field(alias='addDocumentsToCollection')

    class Arguments(BaseModel):
        """Arguments for AddDocumentsToCollection """
        input: AddDocumentsToCollectionInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for AddDocumentsToCollection """
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}\n\nmutation AddDocumentsToCollection($input: AddDocumentsToCollectionInput!) {\n  addDocumentsToCollection(input: $input) {\n    ...Document\n    __typename\n  }\n}'

class GenerateImageMutationGenerateimage(BaseModel):
    """No documentation"""
    typename: Literal['ImageReponse'] = Field(alias='__typename', default='ImageReponse', exclude=True)
    image: str
    model_config = ConfigDict(frozen=True)

class GenerateImageMutation(BaseModel):
    """No documentation found for this operation."""
    generate_image: GenerateImageMutationGenerateimage = Field(alias='generateImage')

    class Arguments(BaseModel):
        """Arguments for GenerateImage """
        input: ImageInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for GenerateImage """
        document = 'mutation GenerateImage($input: ImageInput!) {\n  generateImage(input: $input) {\n    image\n    __typename\n  }\n}'

class SendMutation(BaseModel):
    """No documentation found for this operation."""
    send: Message

    class Arguments(BaseModel):
        """Arguments for Send """
        text: str
        room: ID
        agent_id: str = Field(alias='agentId')
        attach_structures: Optional[List[StructureInput]] = Field(alias='attachStructures', default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for Send """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  agent {\n    id\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nmutation Send($text: String!, $room: ID!, $agentId: String!, $attachStructures: [StructureInput!]) {\n  send(\n    input: {text: $text, room: $room, agentId: $agentId, attachStructures: $attachStructures}\n  ) {\n    ...Message\n    __typename\n  }\n}'

class CreateProviderMutation(BaseModel):
    """No documentation found for this operation."""
    create_provider: Provider = Field(alias='createProvider')

    class Arguments(BaseModel):
        """Arguments for CreateProvider """
        input: ProviderInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for CreateProvider """
        document = 'fragment Provider on Provider {\n  id\n  name\n  models {\n    id\n    modelId\n    features\n    __typename\n  }\n  __typename\n}\n\nmutation CreateProvider($input: ProviderInput!) {\n  createProvider(input: $input) {\n    ...Provider\n    __typename\n  }\n}'

class PullMutationPull(BaseModel):
    """No documentation"""
    typename: Literal['OllamaPullResult'] = Field(alias='__typename', default='OllamaPullResult', exclude=True)
    status: str
    detail: Optional[str] = Field(default=None)
    model_config = ConfigDict(frozen=True)

class PullMutation(BaseModel):
    """No documentation found for this operation."""
    pull: PullMutationPull

    class Arguments(BaseModel):
        """Arguments for Pull """
        input: PullInput
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for Pull """
        document = 'mutation Pull($input: PullInput!) {\n  pull(input: $input) {\n    status\n    detail\n    __typename\n  }\n}'

class CreateRoomMutation(BaseModel):
    """No documentation found for this operation."""
    create_room: Room = Field(alias='createRoom')

    class Arguments(BaseModel):
        """Arguments for CreateRoom """
        title: Optional[str] = Field(default=None)
        description: Optional[str] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for CreateRoom """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nmutation CreateRoom($title: String, $description: String) {\n  createRoom(input: {title: $title, description: $description}) {\n    ...Room\n    __typename\n  }\n}'

class GetChromaCollectionQuery(BaseModel):
    """No documentation found for this operation."""
    chroma_collection: ChromaCollection = Field(alias='chromaCollection')

    class Arguments(BaseModel):
        """Arguments for GetChromaCollection """
        id: ID
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for GetChromaCollection """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nquery GetChromaCollection($id: ID!) {\n  chromaCollection(id: $id) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class SearchChromaCollectionQueryOptions(BaseModel):
    """A collection of documents searchable by string"""
    typename: Literal['ChromaCollection'] = Field(alias='__typename', default='ChromaCollection', exclude=True)
    value: ID
    label: str
    model_config = ConfigDict(frozen=True)

class SearchChromaCollectionQuery(BaseModel):
    """No documentation found for this operation."""
    options: Tuple[SearchChromaCollectionQueryOptions, ...]

    class Arguments(BaseModel):
        """Arguments for SearchChromaCollection """
        search: Optional[str] = Field(default=None)
        values: Optional[List[ID]] = Field(default=None)
        limit: Annotated[Optional[int], GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[Optional[int], GraphQLDefault('0')] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for SearchChromaCollection """
        document = 'query SearchChromaCollection($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: chromaCollections(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: name\n    __typename\n  }\n}'

class ListChromaCollectionsQuery(BaseModel):
    """No documentation found for this operation."""
    chroma_collections: Tuple[ChromaCollection, ...] = Field(alias='chromaCollections')

    class Arguments(BaseModel):
        """Arguments for ListChromaCollections """
        filter: Optional[ChromaCollectionFilter] = Field(default=None)
        order: Optional[List[ChromaCollectionOrder]] = Field(default=None)
        pagination: Optional[OffsetPaginationInput] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for ListChromaCollections """
        document = 'fragment ChromaCollection on ChromaCollection {\n  id\n  name\n  description\n  createdAt\n  __typename\n}\n\nquery ListChromaCollections($filter: ChromaCollectionFilter, $order: [ChromaCollectionOrder!], $pagination: OffsetPaginationInput) {\n  chromaCollections(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...ChromaCollection\n    __typename\n  }\n}'

class QueryDocumentsQuery(BaseModel):
    """No documentation found for this operation."""
    documents: Tuple[Document, ...]

    class Arguments(BaseModel):
        """Arguments for QueryDocuments """
        collection: ID
        query_texts: List[str] = Field(alias='queryTexts')
        n_results: Optional[int] = Field(alias='nResults', default=None)
        where: Optional[Any] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for QueryDocuments """
        document = 'fragment Document on Document {\n  id\n  content\n  metadata\n  distance\n  __typename\n}\n\nquery QueryDocuments($collection: ID!, $queryTexts: [String!]!, $nResults: Int, $where: JSON) {\n  documents(\n    collection: $collection\n    queryTexts: $queryTexts\n    nResults: $nResults\n    where: $where\n  ) {\n    ...Document\n    __typename\n  }\n}'

class GetLLMModelQuery(BaseModel):
    """No documentation found for this operation."""
    llm_model: LLMModel = Field(alias='llmModel')

    class Arguments(BaseModel):
        """Arguments for GetLLMModel """
        id: ID
        model_config = ConfigDict(populate_by_name=True)

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
    options: Tuple[SearchLLMModelsQueryOptions, ...]

    class Arguments(BaseModel):
        """Arguments for SearchLLMModels """
        search: Optional[str] = Field(default=None)
        values: Optional[List[ID]] = Field(default=None)
        limit: Annotated[Optional[int], GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[Optional[int], GraphQLDefault('0')] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for SearchLLMModels """
        document = 'query SearchLLMModels($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: llmModels(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: modelId\n    __typename\n  }\n}'

class ListLLModelsQuery(BaseModel):
    """No documentation found for this operation."""
    llm_models: Tuple[LLMModel, ...] = Field(alias='llmModels')

    class Arguments(BaseModel):
        """Arguments for ListLLModels """
        filter: Optional[LLMModelFilter] = Field(default=None)
        order: Optional[List[LLMModelOrder]] = Field(default=None)
        pagination: Optional[OffsetPaginationInput] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for ListLLModels """
        document = 'fragment LLMModel on LLMModel {\n  id\n  provider {\n    id\n    name\n    __typename\n  }\n  features\n  embedderFor {\n    id\n    name\n    __typename\n  }\n  __typename\n}\n\nquery ListLLModels($filter: LLMModelFilter, $order: [LLMModelOrder!], $pagination: OffsetPaginationInput) {\n  llmModels(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...LLMModel\n    __typename\n  }\n}'

class GetMessageQuery(BaseModel):
    """No documentation found for this operation."""
    message: Message

    class Arguments(BaseModel):
        """Arguments for GetMessage """
        id: ID
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for GetMessage """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nfragment Message on Message {\n  id\n  text\n  agent {\n    id\n    room {\n      id\n      __typename\n    }\n    __typename\n  }\n  room {\n    id\n    title\n    __typename\n  }\n  before {\n    ...ListMessage\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nquery GetMessage($id: ID!) {\n  message(id: $id) {\n    ...Message\n    __typename\n  }\n}'

class SearchMessagesQueryOptions(BaseModel):
    """Message represent the message of an agent on a room"""
    typename: Literal['Message'] = Field(alias='__typename', default='Message', exclude=True)
    value: ID
    label: str
    model_config = ConfigDict(frozen=True)

class SearchMessagesQuery(BaseModel):
    """No documentation found for this operation."""
    options: Tuple[SearchMessagesQueryOptions, ...]

    class Arguments(BaseModel):
        """Arguments for SearchMessages """
        search: Optional[str] = Field(default=None)
        values: Optional[List[ID]] = Field(default=None)
        limit: Annotated[Optional[int], GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[Optional[int], GraphQLDefault('0')] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for SearchMessages """
        document = 'query SearchMessages($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: messages(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: title\n    __typename\n  }\n}'

class ListMessagesQuery(BaseModel):
    """No documentation found for this operation."""
    messages: Tuple[ListMessage, ...]

    class Arguments(BaseModel):
        """Arguments for ListMessages """
        filter: Optional[MessageFilter] = Field(default=None)
        order: Optional[List[MessageOrder]] = Field(default=None)
        pagination: Optional[OffsetPaginationInput] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for ListMessages """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nquery ListMessages($filter: MessageFilter, $order: [MessageOrder!], $pagination: OffsetPaginationInput) {\n  messages(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...ListMessage\n    __typename\n  }\n}'

class GetRoomQuery(BaseModel):
    """No documentation found for this operation."""
    room: Room

    class Arguments(BaseModel):
        """Arguments for GetRoom """
        id: ID
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for GetRoom """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nquery GetRoom($id: ID!) {\n  room(id: $id) {\n    ...Room\n    __typename\n  }\n}'

class SearchRoomsQueryOptions(BaseModel):
    """Room(id, title, description, creator, organization, created_at)"""
    typename: Literal['Room'] = Field(alias='__typename', default='Room', exclude=True)
    value: ID
    label: str
    'The Title of the Room'
    description: str
    model_config = ConfigDict(frozen=True)

class SearchRoomsQuery(BaseModel):
    """No documentation found for this operation."""
    options: Tuple[SearchRoomsQueryOptions, ...]

    class Arguments(BaseModel):
        """Arguments for SearchRooms """
        search: Optional[str] = Field(default=None)
        values: Optional[List[ID]] = Field(default=None)
        limit: Annotated[Optional[int], GraphQLDefault('10')] = Field(default=None)
        offset: Annotated[Optional[int], GraphQLDefault('0')] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for SearchRooms """
        document = 'query SearchRooms($search: String, $values: [ID!], $limit: Int = 10, $offset: Int = 0) {\n  options: rooms(\n    filters: {search: $search, ids: $values}\n    pagination: {limit: $limit, offset: $offset}\n  ) {\n    value: id\n    label: title\n    description: description\n    __typename\n  }\n}'

class ListRoomsQuery(BaseModel):
    """No documentation found for this operation."""
    rooms: Tuple[Room, ...]

    class Arguments(BaseModel):
        """Arguments for ListRooms """
        filter: Optional[RoomFilter] = Field(default=None)
        order: Optional[List[RoomOrder]] = Field(default=None)
        pagination: Optional[OffsetPaginationInput] = Field(default=None)
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for ListRooms """
        document = 'fragment Room on Room {\n  id\n  title\n  description\n  __typename\n}\n\nquery ListRooms($filter: RoomFilter, $order: [RoomOrder!], $pagination: OffsetPaginationInput) {\n  rooms(filters: $filter, ordering: $order, pagination: $pagination) {\n    ...Room\n    __typename\n  }\n}'

class WatchRoomSubscriptionRoom(BaseModel):
    """No documentation"""
    typename: Literal['RoomEvent'] = Field(alias='__typename', default='RoomEvent', exclude=True)
    message: Optional[ListMessage] = Field(default=None)
    model_config = ConfigDict(frozen=True)

class WatchRoomSubscription(BaseModel):
    """No documentation found for this operation."""
    room: WatchRoomSubscriptionRoom

    class Arguments(BaseModel):
        """Arguments for WatchRoom """
        room: ID
        agent_id: ID = Field(alias='agentId')
        model_config = ConfigDict(populate_by_name=True)

    class Meta:
        """Meta class for WatchRoom """
        document = 'fragment ListMessage on Message {\n  id\n  text\n  agent {\n    id\n    __typename\n  }\n  attachedStructures {\n    object\n    identifier\n    __typename\n  }\n  __typename\n}\n\nsubscription WatchRoom($room: ID!, $agentId: ID!) {\n  room(room: $room, agentId: $agentId) {\n    message {\n      ...ListMessage\n      __typename\n    }\n    __typename\n  }\n}'

async def achat(messages: Iterable[ChatMessageInput], model: Union[Optional[IDCoercible], UnsetType]=UNSET, tools: Union[Optional[Iterable[ToolInput]], UnsetType]=UNSET, temperature: Union[Optional[float], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChatResponse:
    """Chat 


Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    messages: A chat message input (required) (list) (required)
    tools: A large language model function call (required) (list)
    temperature: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChatResponse
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['messages'] = messages
    if tools is not UNSET:
        _input['tools'] = tools
    if temperature is not UNSET:
        _input['temperature'] = temperature
    variables['input'] = _input
    return (await aexecute(ChatMutation, variables, rath=rath)).chat

def chat(messages: Iterable[ChatMessageInput], model: Union[Optional[IDCoercible], UnsetType]=UNSET, tools: Union[Optional[Iterable[ToolInput]], UnsetType]=UNSET, temperature: Union[Optional[float], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChatResponse:
    """Chat 


Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    messages: A chat message input (required) (list) (required)
    tools: A large language model function call (required) (list)
    temperature: The `Float` scalar type represents signed double-precision fractional values as specified by [IEEE 754](https://en.wikipedia.org/wiki/IEEE_floating_point).
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChatResponse
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['messages'] = messages
    if tools is not UNSET:
        _input['tools'] = tools
    if temperature is not UNSET:
        _input['temperature'] = temperature
    variables['input'] = _input
    return execute(ChatMutation, variables, rath=rath).chat

async def acreate_collection(name: str, embedder: IDCoercible, description: Union[Optional[str], UnsetType]=UNSET, is_public: Union[Optional[bool], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """CreateCollection 


Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    is_public: The `Boolean` scalar type represents `true` or `false`.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    if is_public is not UNSET:
        _input['isPublic'] = is_public
    variables['input'] = _input
    return (await aexecute(CreateCollectionMutation, variables, rath=rath)).create_collection

def create_collection(name: str, embedder: IDCoercible, description: Union[Optional[str], UnsetType]=UNSET, is_public: Union[Optional[bool], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """CreateCollection 


Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    is_public: The `Boolean` scalar type represents `true` or `false`.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    if is_public is not UNSET:
        _input['isPublic'] = is_public
    variables['input'] = _input
    return execute(CreateCollectionMutation, variables, rath=rath).create_collection

async def aensure_collection(name: str, embedder: IDCoercible, description: Union[Optional[str], UnsetType]=UNSET, is_public: Union[Optional[bool], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """EnsureCollection 


Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    is_public: The `Boolean` scalar type represents `true` or `false`.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    if is_public is not UNSET:
        _input['isPublic'] = is_public
    variables['input'] = _input
    return (await aexecute(EnsureCollectionMutation, variables, rath=rath)).ensure_collection

def ensure_collection(name: str, embedder: IDCoercible, description: Union[Optional[str], UnsetType]=UNSET, is_public: Union[Optional[bool], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """EnsureCollection 


Args:
    name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    embedder: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
    is_public: The `Boolean` scalar type represents `true` or `false`.
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['name'] = name
    _input['embedder'] = embedder
    if description is not UNSET:
        _input['description'] = description
    if is_public is not UNSET:
        _input['isPublic'] = is_public
    variables['input'] = _input
    return execute(EnsureCollectionMutation, variables, rath=rath).ensure_collection

async def aadd_documents_to_collection(collection: IDCoercible, documents: Iterable[DocumentInput], rath: Optional[AlpakaRath]=None) -> Tuple[Document, ...]:
    """AddDocumentsToCollection 


Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    documents: A document to put into the vector database (required) (list) (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Document]
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['collection'] = collection
    _input['documents'] = documents
    variables['input'] = _input
    return (await aexecute(AddDocumentsToCollectionMutation, variables, rath=rath)).add_documents_to_collection

def add_documents_to_collection(collection: IDCoercible, documents: Iterable[DocumentInput], rath: Optional[AlpakaRath]=None) -> Tuple[Document, ...]:
    """AddDocumentsToCollection 


Args:
    collection: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID. (required)
    documents: A document to put into the vector database (required) (list) (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Document]
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['collection'] = collection
    _input['documents'] = documents
    variables['input'] = _input
    return execute(AddDocumentsToCollectionMutation, variables, rath=rath).add_documents_to_collection

async def agenerate_image(description: str, model: Union[Optional[IDCoercible], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> GenerateImageMutationGenerateimage:
    """GenerateImage 


Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    GenerateImageMutationGenerateimage
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['description'] = description
    variables['input'] = _input
    return (await aexecute(GenerateImageMutation, variables, rath=rath)).generate_image

def generate_image(description: str, model: Union[Optional[IDCoercible], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> GenerateImageMutationGenerateimage:
    """GenerateImage 


Args:
    model: The `ID` scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as `"4"`) or integer (such as `4`) input value will be accepted as an ID.
    description: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    GenerateImageMutationGenerateimage
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    if model is not UNSET:
        _input['model'] = model
    _input['description'] = description
    variables['input'] = _input
    return execute(GenerateImageMutation, variables, rath=rath).generate_image

async def asend(text: str, room: ID, agent_id: str, attach_structures: Union[Optional[List[StructureInput]], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Message:
    """Send 


Args:
    text (str): No description
    room (ID): No description
    agent_id (str): No description
    attach_structures (Optional[List[StructureInput]], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: Dict[str, Any] = {}
    variables['text'] = text
    variables['room'] = room
    variables['agentId'] = agent_id
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return (await aexecute(SendMutation, variables, rath=rath)).send

def send(text: str, room: ID, agent_id: str, attach_structures: Union[Optional[List[StructureInput]], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Message:
    """Send 


Args:
    text (str): No description
    room (ID): No description
    agent_id (str): No description
    attach_structures (Optional[List[StructureInput]], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: Dict[str, Any] = {}
    variables['text'] = text
    variables['room'] = room
    variables['agentId'] = agent_id
    if attach_structures is not UNSET:
        variables['attachStructures'] = attach_structures
    return execute(SendMutation, variables, rath=rath).send

async def acreate_provider(kind: ProviderKind, description: Union[Optional[str], UnsetType]=UNSET, name: Union[Optional[str], UnsetType]=UNSET, api_key: Union[Optional[str], UnsetType]=UNSET, api_base: Union[Optional[str], UnsetType]=UNSET, additional_config: Union[Optional[Any], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Provider:
    """CreateProvider 


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
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
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

def create_provider(kind: ProviderKind, description: Union[Optional[str], UnsetType]=UNSET, name: Union[Optional[str], UnsetType]=UNSET, api_key: Union[Optional[str], UnsetType]=UNSET, api_base: Union[Optional[str], UnsetType]=UNSET, additional_config: Union[Optional[Any], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Provider:
    """CreateProvider 


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
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
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

async def apull(model_name: str, rath: Optional[AlpakaRath]=None) -> PullMutationPull:
    """Pull 


Args:
    model_name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    PullMutationPull
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['modelName'] = model_name
    variables['input'] = _input
    return (await aexecute(PullMutation, variables, rath=rath)).pull

def pull(model_name: str, rath: Optional[AlpakaRath]=None) -> PullMutationPull:
    """Pull 


Args:
    model_name: The `String` scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text. (required)
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    PullMutationPull
"""
    variables: Dict[str, Any] = {}
    _input: Dict[str, Any] = {}
    _input['modelName'] = model_name
    variables['input'] = _input
    return execute(PullMutation, variables, rath=rath).pull

async def acreate_room(title: Union[Optional[str], UnsetType]=UNSET, description: Union[Optional[str], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Room:
    """CreateRoom 


Args:
    title (Optional[str], optional): No description. 
    description (Optional[str], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: Dict[str, Any] = {}
    if title is not UNSET:
        variables['title'] = title
    if description is not UNSET:
        variables['description'] = description
    return (await aexecute(CreateRoomMutation, variables, rath=rath)).create_room

def create_room(title: Union[Optional[str], UnsetType]=UNSET, description: Union[Optional[str], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Room:
    """CreateRoom 


Args:
    title (Optional[str], optional): No description. 
    description (Optional[str], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: Dict[str, Any] = {}
    if title is not UNSET:
        variables['title'] = title
    if description is not UNSET:
        variables['description'] = description
    return execute(CreateRoomMutation, variables, rath=rath).create_room

async def aget_chroma_collection(id: ID, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """GetChromaCollection 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetChromaCollectionQuery, variables, rath=rath)).chroma_collection

def get_chroma_collection(id: ID, rath: Optional[AlpakaRath]=None) -> ChromaCollection:
    """GetChromaCollection 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    ChromaCollection
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return execute(GetChromaCollectionQuery, variables, rath=rath).chroma_collection

async def asearch_chroma_collection(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchChromaCollectionQueryOptions, ...]:
    """SearchChromaCollection 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchChromaCollectionQueryChromacollections]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchChromaCollectionQuery, variables, rath=rath)).options

def search_chroma_collection(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchChromaCollectionQueryOptions, ...]:
    """SearchChromaCollection 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchChromaCollectionQueryChromacollections]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchChromaCollectionQuery, variables, rath=rath).options

async def alist_chroma_collections(filter: Union[Optional[ChromaCollectionFilter], UnsetType]=UNSET, order: Union[Optional[List[ChromaCollectionOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[ChromaCollection, ...]:
    """ListChromaCollections 


Args:
    filter (Optional[ChromaCollectionFilter], optional): No description. 
    order (Optional[List[ChromaCollectionOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[ChromaCollection]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListChromaCollectionsQuery, variables, rath=rath)).chroma_collections

def list_chroma_collections(filter: Union[Optional[ChromaCollectionFilter], UnsetType]=UNSET, order: Union[Optional[List[ChromaCollectionOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[ChromaCollection, ...]:
    """ListChromaCollections 


Args:
    filter (Optional[ChromaCollectionFilter], optional): No description. 
    order (Optional[List[ChromaCollectionOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[ChromaCollection]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListChromaCollectionsQuery, variables, rath=rath).chroma_collections

async def aquery_documents(collection: ID, query_texts: List[str], n_results: Union[Optional[int], UnsetType]=UNSET, where: Union[Optional[Any], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[Document, ...]:
    """QueryDocuments 


Args:
    collection (ID): No description
    query_texts (List[str]): No description
    n_results (Optional[int], optional): No description. 
    where (Optional[Any], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Document]
"""
    variables: Dict[str, Any] = {}
    variables['collection'] = collection
    variables['queryTexts'] = query_texts
    if n_results is not UNSET:
        variables['nResults'] = n_results
    if where is not UNSET:
        variables['where'] = where
    return (await aexecute(QueryDocumentsQuery, variables, rath=rath)).documents

def query_documents(collection: ID, query_texts: List[str], n_results: Union[Optional[int], UnsetType]=UNSET, where: Union[Optional[Any], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[Document, ...]:
    """QueryDocuments 


Args:
    collection (ID): No description
    query_texts (List[str]): No description
    n_results (Optional[int], optional): No description. 
    where (Optional[Any], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Document]
"""
    variables: Dict[str, Any] = {}
    variables['collection'] = collection
    variables['queryTexts'] = query_texts
    if n_results is not UNSET:
        variables['nResults'] = n_results
    if where is not UNSET:
        variables['where'] = where
    return execute(QueryDocumentsQuery, variables, rath=rath).documents

async def aget_llm_model(id: ID, rath: Optional[AlpakaRath]=None) -> LLMModel:
    """GetLLMModel 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    LLMModel
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetLLMModelQuery, variables, rath=rath)).llm_model

def get_llm_model(id: ID, rath: Optional[AlpakaRath]=None) -> LLMModel:
    """GetLLMModel 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    LLMModel
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return execute(GetLLMModelQuery, variables, rath=rath).llm_model

async def asearch_llm_models(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchLLMModelsQueryOptions, ...]:
    """SearchLLMModels 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchLLMModelsQueryLlmmodels]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchLLMModelsQuery, variables, rath=rath)).options

def search_llm_models(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchLLMModelsQueryOptions, ...]:
    """SearchLLMModels 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchLLMModelsQueryLlmmodels]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchLLMModelsQuery, variables, rath=rath).options

async def alist_ll_models(filter: Union[Optional[LLMModelFilter], UnsetType]=UNSET, order: Union[Optional[List[LLMModelOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[LLMModel, ...]:
    """ListLLModels 


Args:
    filter (Optional[LLMModelFilter], optional): No description. 
    order (Optional[List[LLMModelOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[LLMModel]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListLLModelsQuery, variables, rath=rath)).llm_models

def list_ll_models(filter: Union[Optional[LLMModelFilter], UnsetType]=UNSET, order: Union[Optional[List[LLMModelOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[LLMModel, ...]:
    """ListLLModels 


Args:
    filter (Optional[LLMModelFilter], optional): No description. 
    order (Optional[List[LLMModelOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[LLMModel]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListLLModelsQuery, variables, rath=rath).llm_models

async def aget_message(id: ID, rath: Optional[AlpakaRath]=None) -> Message:
    """GetMessage 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetMessageQuery, variables, rath=rath)).message

def get_message(id: ID, rath: Optional[AlpakaRath]=None) -> Message:
    """GetMessage 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Message
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return execute(GetMessageQuery, variables, rath=rath).message

async def asearch_messages(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchMessagesQueryOptions, ...]:
    """SearchMessages 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchMessagesQueryMessages]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchMessagesQuery, variables, rath=rath)).options

def search_messages(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchMessagesQueryOptions, ...]:
    """SearchMessages 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchMessagesQueryMessages]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchMessagesQuery, variables, rath=rath).options

async def alist_messages(filter: Union[Optional[MessageFilter], UnsetType]=UNSET, order: Union[Optional[List[MessageOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[ListMessage, ...]:
    """ListMessages 


Args:
    filter (Optional[MessageFilter], optional): No description. 
    order (Optional[List[MessageOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[ListMessage]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListMessagesQuery, variables, rath=rath)).messages

def list_messages(filter: Union[Optional[MessageFilter], UnsetType]=UNSET, order: Union[Optional[List[MessageOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[ListMessage, ...]:
    """ListMessages 


Args:
    filter (Optional[MessageFilter], optional): No description. 
    order (Optional[List[MessageOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[ListMessage]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListMessagesQuery, variables, rath=rath).messages

async def aget_room(id: ID, rath: Optional[AlpakaRath]=None) -> Room:
    """GetRoom 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return (await aexecute(GetRoomQuery, variables, rath=rath)).room

def get_room(id: ID, rath: Optional[AlpakaRath]=None) -> Room:
    """GetRoom 


Args:
    id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    Room
"""
    variables: Dict[str, Any] = {}
    variables['id'] = id
    return execute(GetRoomQuery, variables, rath=rath).room

async def asearch_rooms(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchRoomsQueryOptions, ...]:
    """SearchRooms 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchRoomsQueryRooms]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return (await aexecute(SearchRoomsQuery, variables, rath=rath)).options

def search_rooms(search: Union[Optional[str], UnsetType]=UNSET, values: Union[Optional[List[ID]], UnsetType]=UNSET, limit: Union[Optional[int], UnsetType]=UNSET, offset: Union[Optional[int], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[SearchRoomsQueryOptions, ...]:
    """SearchRooms 


Args:
    search (Optional[str], optional): No description. 
    values (Optional[List[ID]], optional): No description. 
    limit (Optional[int], optional): No description. Defaults to 10
    offset (Optional[int], optional): No description. Defaults to 0
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[SearchRoomsQueryRooms]
"""
    variables: Dict[str, Any] = {}
    if search is not UNSET:
        variables['search'] = search
    if values is not UNSET:
        variables['values'] = values
    if limit is not UNSET:
        variables['limit'] = limit
    if offset is not UNSET:
        variables['offset'] = offset
    return execute(SearchRoomsQuery, variables, rath=rath).options

async def alist_rooms(filter: Union[Optional[RoomFilter], UnsetType]=UNSET, order: Union[Optional[List[RoomOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[Room, ...]:
    """ListRooms 


Args:
    filter (Optional[RoomFilter], optional): No description. 
    order (Optional[List[RoomOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Room]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return (await aexecute(ListRoomsQuery, variables, rath=rath)).rooms

def list_rooms(filter: Union[Optional[RoomFilter], UnsetType]=UNSET, order: Union[Optional[List[RoomOrder]], UnsetType]=UNSET, pagination: Union[Optional[OffsetPaginationInput], UnsetType]=UNSET, rath: Optional[AlpakaRath]=None) -> Tuple[Room, ...]:
    """ListRooms 


Args:
    filter (Optional[RoomFilter], optional): No description. 
    order (Optional[List[RoomOrder]], optional): No description. 
    pagination (Optional[OffsetPaginationInput], optional): No description. 
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    List[Room]
"""
    variables: Dict[str, Any] = {}
    if filter is not UNSET:
        variables['filter'] = filter
    if order is not UNSET:
        variables['order'] = order
    if pagination is not UNSET:
        variables['pagination'] = pagination
    return execute(ListRoomsQuery, variables, rath=rath).rooms

async def awatch_room(room: ID, agent_id: ID, rath: Optional[AlpakaRath]=None) -> AsyncIterator[WatchRoomSubscriptionRoom]:
    """WatchRoom 


Args:
    room (ID): No description
    agent_id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    WatchRoomSubscriptionRoom
"""
    variables: Dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
    async for event in asubscribe(WatchRoomSubscription, variables, rath=rath):
        yield event.room

def watch_room(room: ID, agent_id: ID, rath: Optional[AlpakaRath]=None) -> Iterator[WatchRoomSubscriptionRoom]:
    """WatchRoom 


Args:
    room (ID): No description
    agent_id (ID): No description
    rath (alpaka.rath.AlpakaRath, optional): The client we want to use (defaults to the currently active client)

Returns:
    WatchRoomSubscriptionRoom
"""
    variables: Dict[str, Any] = {}
    variables['room'] = room
    variables['agentId'] = agent_id
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