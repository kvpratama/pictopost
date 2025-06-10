from langgraph.constants import Send
import os
import logging
import time
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import GraphState, ImageProcessingInputState
from PIL import Image
from prompts import load_prompt
import base64
import pillow_heif  # Add HEIF support
# Register HEIF plugin with PIL
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)


def initiate_image_processing(state: GraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    logger.info(f"Initiating image processing")
    
    return [Send("image_processing", {"image_path": image_path, "max_size": state["max_size"]}) for image_path in state["image_paths"]]


def resize_image(state: ImageProcessingInputState):
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


def describe_image(state: ImageProcessingInputState):
    """
    
    """
    logger.info(f"Describing image: {state['resized_images'][0]}")

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
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
    

def human_feedback(state: GraphState):
    """ No-op node that should be interrupted on """
    logger.info("Human feedback")
    pass