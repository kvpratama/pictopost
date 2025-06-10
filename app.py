import streamlit as st
import tempfile
import os
from langgraph_client import LangGraphLocalClient
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Initializing LangGraphLocalClient...")
st.session_state["client"] = LangGraphLocalClient()
st.session_state["response"] = None

st.title("PicToPost")

uploaded_files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg", "heic", "heif"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"{len(uploaded_files)} image(s) selected.")
    if st.button("Resize Images"):
        paths = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                paths.append(tmp_path)
            
        logger.info("Generating analysts...")
        with st.spinner("Generating analysts..."):
            # Prepare input data for backend
            input_data = {
                "image_paths": paths,
                "max_size": 800,
            }
            st.session_state["response"] = st.session_state["client"].run_graph(input_data=input_data)
            # st.rerun()
        st.success(f"Resized {len(st.session_state["response"]["resized_images"])}) image(s). Thumbnails below:")
        cols = st.columns(len(st.session_state["response"]["resized_images"]))
        for idx, img in enumerate(st.session_state["response"]["resized_images"]):
            thumb = Image.open(img)
            thumb.thumbnail((128, 128))
            cols[idx].image(thumb, caption=f"Image {idx+1}", use_container_width=True)
else:
    st.info("Please upload one or more images.")
