from langgraph.graph import START, END, StateGraph
from state import GraphState, ImageProcessingInputState, ImageProcessingOutputState
from nodes import initiate_image_processing, human_feedback, resize_image, describe_image
from langgraph.checkpoint.memory import MemorySaver

def get_graph():
    builder = StateGraph(GraphState)
    builder.add_node("initiate_image_processing", initiate_image_processing)
    builder.add_node("image_processing", get_image_processing_builder().compile())
    builder.add_node("human_feedback", human_feedback)

    # Logic
    builder.add_conditional_edges(START, initiate_image_processing, ["image_processing"])
    builder.add_edge("image_processing", "human_feedback")
    builder.add_edge("human_feedback", END)

    # Compile
    memory = MemorySaver()
    graph_memory = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    return graph_memory


def get_image_processing_builder():
    builder = StateGraph(input=ImageProcessingInputState, output=ImageProcessingOutputState)
    builder.add_node("resize_image", resize_image)
    builder.add_node("describe_image", describe_image)

    # Logic
    builder.add_edge(START, "resize_image")
    builder.add_edge("resize_image", "describe_image")
    builder.add_edge("describe_image", END)

    return builder