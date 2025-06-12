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
    blog_post: str
    translated_content: str
    localized_content: str

class GraphStateInput(MessagesState):
    image_paths: Annotated[List[str], add]
    max_size: int

class GraphStateOutput(MessagesState):
    resized_images: Annotated[List[str], add]
    image_descriptions: Annotated[List[str], add]
    additional_context: str
    blog_post: str
    translated_content: str
    localized_content: str

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
    blog_post: str

class WritingStateInput(MessagesState):
    image_descriptions: Annotated[List[str], add]
    additional_context: str

class WritingStateOutput(MessagesState):
    blog_post: str