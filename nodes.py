from langgraph.constants import Send
import os
import logging
import time
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from state import GraphState, ImageProcessingState
from PIL import Image
from prompts import load_prompt
import base64
import pillow_heif  # Add HEIF support
# Register HEIF plugin with PIL
pillow_heif.register_heif_opener()
from llm_model import get_gemma27b_llm, get_gemma12b_llm
from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def initiate_image_processing(state: GraphState, config: dict):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    logger.info(f"Initiating image processing")
    
    return [Send("image_processing", {"image_path": image_path, "max_size": state["max_size"]}) for image_path in state["image_paths"]]


def resize_image(state: ImageProcessingState, config: dict):
    """
    Resize an image so that its longest side does not exceed a specified maximum size.

    This function reads an image from the path specified in the `state` dictionary, 
    scales it proportionally so that its longest dimension matches `state["max_size"]`,
    and saves the resized image with "_resized" appended to the original filename.

    Args:
        state (ImageProcessingState): A dictionary-like object containing:
            - "image_path" (str): Path to the input image file.
            - "max_size" (int): The maximum allowed size for the image's longest side.

    Returns:
        dict: A dictionary containing the path to the resized image under the key "resized_images".
    """

    logger.info(f"Resizing image: {state['image_path']}")
    image = Image.open(state["image_path"])
    original_width, original_height = image.size

    # Determine scale factor
    if original_width > original_height:
        scale = state["max_size"] / float(original_width)
    else:
        scale = state["max_size"] / float(original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # save resized image using original filename
    resized_image.save(os.path.splitext(state["image_path"])[0] + "_resized.jpg")
    logger.info(f"Resized image: {os.path.splitext(state["image_path"])[0] + "_resized.jpg"}")
    return {"resized_images": [os.path.splitext(state["image_path"])[0] + "_resized.jpg"]}


def describe_image(state: ImageProcessingState, config: dict):
    """
    
    """
    logger.info(f"Describing image: {state['resized_images'][0]}")

    # Initialize the Gemini model
    google_api_key = config["configurable"]["google_api_key"]
    llm = get_gemma27b_llm(google_api_key=google_api_key)
    # Load and encode local image
    with open(state["resized_images"][0], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    image_description_instructions = load_prompt("image_description_instructions")
    message_local = HumanMessage(
        content=[
            {"type": "text", "text": image_description_instructions},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
        ]
    )

    response = llm.invoke([message_local])
    logger.info(f"Image description: {response.content}")
    return {"image_descriptions": [response.content]}
    

def human_feedback(state: GraphState, config: dict):
    """ No-op node that should be interrupted on """
    logger.info("Human feedback")
    pass


def write_blog_post(state: GraphState, config: dict):
    """ """
    logger.info("Writing blog post")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Writing blog post...*\n"})
    images_description = "\n".join(state["image_descriptions"])
    additional_context = state["additional_context"]
    google_api_key = config["configurable"]["google_api_key"]

    blogger_instruction = load_prompt("blogger_instruction")
    system_message = blogger_instruction.format(images_description=images_description, additional_context=additional_context)
    llm = get_gemma12b_llm(google_api_key=google_api_key)
    response = llm.invoke([HumanMessage(content=system_message)])
    response.name = "writer"

    logger.info(f"Blog post: {response.content}")
    stream_writer({"custom_key": "*Post written*:\n" + response.content})
    return {"blog_post": response.content, "messages": [response]}


def editor_feedback(state: GraphState, config: dict):
    """ No-op node that should be interrupted on """
    logger.info("Editor feedback")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Editor is reading content...*\n"})
    google_api_key = config["configurable"]["google_api_key"]

    editor_instruction = load_prompt("editor_instruction")
    system_message = editor_instruction.format(blog_post=state["blog_post"])
    llm = get_gemma27b_llm(google_api_key=google_api_key)
    response = llm.invoke([HumanMessage(content=system_message)])

    logger.info(f"Editor feedback: {response.content}")
    editor_message = HumanMessage(content=response.content)
    editor_message.name = "editor"
    stream_writer({"custom_key": "*Editor's feedback*:\n" + editor_message.content})
    return {"messages": [editor_message]}


def refine_blog_post(state: GraphState, config: dict):
    """ """
    logger.info("Refining blog post")
    google_api_key = config["configurable"]["google_api_key"]
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Refining content based on editor's feedback...*\n"})

    llm = get_gemma12b_llm(google_api_key=google_api_key)
    instruction = HumanMessage(content="Refine the blog post based on the editor's feedback")
    response = llm.invoke(state["messages"] + [instruction])
    response.name = "refiner"

    logger.info(f"Refined content: {response.content}")
    stream_writer({"custom_key": "*Refined content*:\n" + response.content})
    return {"blog_post": response.content, "messages": [response]}


def writing_flow_control(state: GraphState, config: dict):
    """ """
    logger.info("Writing flow control")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Deciding on next step...*\n"})

    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == "refiner"]
        )

    logger.info(f"Number of responses: {num_responses}/{max_num_turns}")
    stream_writer({"custom_key": f"*Number of responses: {num_responses}/{max_num_turns}*\n"})

    if num_responses >= max_num_turns:
        logger.info("Reached maximum number of turns")
        stream_writer({"custom_key": "*Reached maximum number of turns*\n *Finishing Writing Process...*\n"})
        return "human_content_feedback"

    logger.info("Continuing Writing Process")
    stream_writer({"custom_key": "*Continuing Writing Process...*\n"})
    return "editor_feedback"


def translate_content(state: GraphState, config: dict):
    """ """
    logger.info("Translating content")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Translating content...*\n"})

    google_api_key = config["configurable"]["google_api_key"]

    translator_instruction = load_prompt("translator_instruction")
    system_message = translator_instruction.format(content=state["blog_post"], target_language="Indonesian")
    llm = get_gemma27b_llm(google_api_key=google_api_key)
    instruction = HumanMessage(content=system_message)
    response = llm.invoke([instruction])
    response.name = "translator"

    logger.info(f"Translated content: {response.content}")
    stream_writer({"custom_key": "*Translated content*:\n" + response.content})
    return {"translated_content": response.content}


def localize_content(state: GraphState, config: dict):
    """ """
    logger.info("Localizing content")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "*Localizing content...*\n"})

    google_api_key = config["configurable"]["google_api_key"]

    localized_instruction = load_prompt("localizer_instruction")
    system_message = localized_instruction.format(translated_content=state["translated_content"], target_locale="Indonesian")
    llm = get_gemma27b_llm(google_api_key=google_api_key)
    instruction = HumanMessage(content=system_message)
    response = llm.invoke([instruction])
    response.name = "localizer"

    logger.info(f"Localized content: {response.content}")
    stream_writer({"custom_key": "*Localized content*:\n" + response.content})
    return {"localized_content": response.content}