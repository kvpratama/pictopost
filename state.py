from langgraph.graph import MessagesState
from typing import TypedDict
from operator import add
from typing import List, Annotated

class GraphState(MessagesState):
    image_paths: Annotated[List[str], add]
    max_size: int
    resized_images: Annotated[List[str], add]
    image_descriptions: Annotated[List[str], add]
    additional_context: str
    content: str
    caption: str
    target_language: str
    localized_content: Annotated[List[str], add]

class GraphStateInput(MessagesState):
    image_paths: Annotated[List[str], add]
    max_size: int

class GraphStateOutput(MessagesState):
    resized_images: Annotated[List[str], add]
    image_descriptions: Annotated[List[str], add]
    additional_context: str
    content: str
    caption: str
    localized_content: Annotated[List[str], add]


class ImageProcessingState(TypedDict):
    image_path: str
    max_size: int
    resized_images: List[str]

class ImageProcessingStateInput(TypedDict):
    image_path: str
    max_size: int

class ImageProcessingStateOutput(TypedDict):
    resized_images: List[str]
    image_descriptions: List[str]


class WritingState(MessagesState):
    image_descriptions: Annotated[List[str], add]
    additional_context: str
    content: str
    caption: str

class WritingStateInput(MessagesState):
    image_descriptions: Annotated[List[str], add]
    additional_context: str

class WritingStateOutput(MessagesState):
    content: str
    caption: str


class TranslationState(MessagesState):
    content: str
    target_language: str
    translated_content: str
    localized_content: Annotated[List[str], add]

class TranslationStateInput(MessagesState):
    content: str
    target_language: str

class TranslationStateOutput(MessagesState):
    localized_content: Annotated[List[str], add]