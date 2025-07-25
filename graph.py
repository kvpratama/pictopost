from langgraph.graph import START, END, StateGraph
from state import *
from nodes import *
from langgraph.checkpoint.memory import MemorySaver
from configuration import ConfigSchema


def get_graph():
    builder = StateGraph(
        GraphState,
        input=GraphStateInput,
        output=GraphStateOutput,
        config_schema=ConfigSchema,
    )
    builder.add_node("image_processing", get_image_processing_builder().compile())
    builder.add_node("human_context", human_feedback)
    builder.add_node("writing_graph", get_writing_graph_builder().compile())
    builder.add_node("human_content_feedback", human_feedback)
    builder.add_node("translation_graph", get_translation_graph_builder().compile())

    # Logic
    builder.add_conditional_edges(
        START, initiate_image_processing, ["image_processing"]
    )
    builder.add_edge("image_processing", "human_context")
    builder.add_edge("human_context", "writing_graph")
    builder.add_edge("writing_graph", "human_content_feedback")
    builder.add_conditional_edges(
        "human_content_feedback", initiate_translation, ["translation_graph"]
    )
    builder.add_edge("translation_graph", END)

    # Compile
    memory = MemorySaver()
    graph_memory = builder.compile(
        interrupt_before=["human_content_feedback"],
        interrupt_after=["human_context"],
        checkpointer=memory,
    )
    return graph_memory


def get_image_processing_builder():
    builder = StateGraph(
        ImageProcessingState,
        input=ImageProcessingStateInput,
        output=ImageProcessingStateOutput,
        config_schema=ConfigSchema,
    )
    builder.add_node("resize_image", resize_image)
    builder.add_node("describe_image", describe_image)

    # Logic
    builder.add_edge(START, "resize_image")
    builder.add_edge("resize_image", "describe_image")
    builder.add_edge("describe_image", END)

    return builder


def get_writing_graph_builder():
    builder = StateGraph(
        WritingState,
        input=WritingStateInput,
        output=WritingStateOutput,
        config_schema=ConfigSchema,
    )
    builder.add_node("write_blog_post", write_blog_post)
    builder.add_node("editor_feedback", editor_feedback)
    builder.add_node("refine_blog_post", refine_blog_post)
    builder.add_node("generate_caption", generate_caption)

    # Logic
    builder.add_edge(START, "write_blog_post")
    builder.add_edge("write_blog_post", "editor_feedback")
    builder.add_edge("editor_feedback", "refine_blog_post")
    builder.add_conditional_edges(
        "refine_blog_post",
        writing_flow_control,
        ["editor_feedback", "generate_caption"],
    )
    builder.add_edge("generate_caption", END)

    return builder


def get_translation_graph_builder():
    builder = StateGraph(
        TranslationState,
        input=TranslationStateInput,
        output=TranslationStateOutput,
        config_schema=ConfigSchema,
    )
    builder.add_node("translate_content", translate_content)
    builder.add_node("localize_content", localize_content)

    # Logic
    builder.add_edge(START, "translate_content")
    builder.add_edge("translate_content", "localize_content")
    builder.add_edge("localize_content", END)

    return builder
