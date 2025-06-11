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

class ImageProcessingInputState(TypedDict):
    image_path: str
    max_size: int
    resized_images: List[str]

class ImageProcessingOutputState(TypedDict):
    resized_images: List[str]
    image_descriptions: List[str]