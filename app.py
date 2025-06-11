import streamlit as st
import tempfile
import os
from langgraph_client import LangGraphLocalClient
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "client" not in st.session_state:
    logger.info("Initializing LangGraphLocalClient...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    st.session_state["client"] = LangGraphLocalClient(google_api_key)
    st.session_state["response"] = None

st.title("PicToPost")

if not st.session_state["response"]:
    uploaded_files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg", "heic", "heif"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} image(s) selected.")
        if st.button("Process Images"):
            logger.info("Processing images...")
            with st.spinner("Processing images..."):
                paths = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.read())
                        paths.append(tmp.name)
            
                # Prepare input data for backend
                input_data = {
                    "image_paths": paths,
                    "max_size": 800,
                }
                st.session_state["response"] = st.session_state["client"].run_graph(input_data=input_data)
            st.rerun()
    else:
        st.info("Please upload one or more images.")

else:
    st.success(f"Processed {len(st.session_state["response"]["resized_images"])} image(s)")
    resized_images = st.session_state["response"]["resized_images"]
    image_descriptions = st.session_state["response"]["image_descriptions"]
    for idx, (img, desc) in enumerate(zip(resized_images, image_descriptions)):
        col1, col2 = st.columns([1, 2])
        thumb = Image.open(img)
        thumb.thumbnail((256, 256))
        col1.image(thumb, caption=f"Image {idx+1}", use_container_width=True)
        col2.markdown(desc)
    
    if "blog_post" not in st.session_state["response"]:
        text_area = st.text_area("Optional: Additional Context", "")
        if st.button("Start Writing Process"):
            logger.info("Starting writing process...")
            input_data = {
                "additional_context": text_area,
        }
            with st.spinner("Start writing process..."):
                # st.session_state["response"] = st.session_state["client"].run_graph_resume(input_data=input_data)
                with st.container(height=300):
                    st.write_stream(st.session_state["client"].run_graph_stream(input_data=input_data, stream_mode="custom"))
            st.session_state["response"] = st.session_state["client"].get_state()
            # st.rerun()
    if "blog_post" in st.session_state["response"] and st.session_state["response"]["blog_post"]:
        st.markdown(st.session_state["response"]["blog_post"])
