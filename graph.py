from langgraph.graph import START, END, StateGraph
from state import GraphState, ImageProcessingInputState, ImageProcessingOutputState
from nodes import initiate_image_processing, human_feedback, resize_image, describe_image, write_blog_post, editor_feedback, refine_blog_post, writing_flow_control, translate_content, localize_content
from langgraph.checkpoint.memory import MemorySaver
from configuration import ConfigSchema

def get_graph():
    builder = StateGraph(GraphState, config_schema=ConfigSchema)
    builder.add_node("image_processing", get_image_processing_builder().compile())
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("write_blog_post", write_blog_post)
    builder.add_node("editor_feedback", editor_feedback)
    builder.add_node("refine_blog_post", refine_blog_post)
    builder.add_node("human_content_feedback", human_feedback)
    builder.add_node("translate_content", translate_content)
    builder.add_node("localize_content", localize_content)

    # Logic
    builder.add_conditional_edges(START, initiate_image_processing, ["image_processing"])
    builder.add_edge("image_processing", "human_feedback")
    builder.add_edge("human_feedback", "write_blog_post")
    builder.add_edge("write_blog_post", "editor_feedback")
    builder.add_edge("editor_feedback", "refine_blog_post")
    builder.add_conditional_edges("refine_blog_post", writing_flow_control, ["editor_feedback", "human_content_feedback"])
    builder.add_edge("human_content_feedback", "translate_content")
    builder.add_edge("translate_content", "localize_content")
    builder.add_edge("localize_content", END)

    # Compile
    memory = MemorySaver()
    graph_memory = builder.compile(interrupt_after=['human_feedback', 'human_content_feedback'], checkpointer=memory)
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