from langgraph.graph import MessagesState
from pydantic import BaseModel
from operator import add
from typing import List, Annotated

class GraphState(MessagesState):
    image_paths: Annotated[List[str], add]
    max_size: int
    resized_images: Annotated[List[str], add]
    image_descriptions: Annotated[List[str], add]

class ResizeImageState(BaseModel):
    image_path: str
    max_size: int
    resized_images: List[str]