from langgraph.constants import Send
import os
import logging
import time
from state import GraphState, ResizeImageState
from PIL import Image
import pillow_heif  # Add HEIF support
# Register HEIF plugin with PIL
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)


def initiate_image_resize(state: GraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    logger.info(f"Initiating image resize")
    
    return [Send("resize_image", {"image_path": image_path, "max_size": state["max_size"]}) for image_path in state["image_paths"]]


def resize_image(state: ResizeImageState):
    """
    Resize an image so that its longest side does not exceed a specified maximum size.

    This function reads an image from the path specified in the `state` dictionary, 
    scales it proportionally so that its longest dimension matches `state["max_size"]`,
    and saves the resized image with "_resized" appended to the original filename.

    Args:
        state (ResizeImageState): A dictionary-like object containing:
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


def human_feedback(state: GraphState):
    """ No-op node that should be interrupted on """
    logger.info("Human feedback")
    pass