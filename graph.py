from langgraph.graph import START, END, StateGraph
from state import GraphState
from nodes import initiate_image_resize, human_feedback, resize_image
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(GraphState)
builder.add_node("initiate_image_resize", initiate_image_resize)
builder.add_node("resize_image", resize_image)
builder.add_node("human_feedback", human_feedback)

# Logic
builder.add_conditional_edges(START, initiate_image_resize, ["resize_image"])
builder.add_edge("resize_image", "human_feedback")
builder.add_edge("human_feedback", END)

# Compile
memory = MemorySaver()
graph_memory = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)