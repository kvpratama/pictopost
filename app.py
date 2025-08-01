import streamlit as st
import tempfile
import os
from langgraph_client import LangGraphLocalClient
import logging
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PicToPost",
    page_icon="📜",
)

if "client" not in st.session_state:
    logger.info("Initializing LangGraphLocalClient...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    st.session_state["client"] = LangGraphLocalClient(google_api_key)
    st.session_state["response"] = None
    st.session_state["images"] = []
    st.session_state["writing_in_progress"] = False

st.title("PicToPost")

if not st.session_state["response"]:
    uploaded_files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg", "heic", "heif"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} image(s) selected.")
        if st.button("Process Images"):
            logger.info("Processing images...")
            with st.spinner("Processing images..."):
                paths = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp:
                        logger.info(f"Saving temporary file: {tmp.name}")
                        tmp.write(uploaded_file.read())
                        paths.append(tmp.name)

                # Prepare input data for backend
                input_data = {
                    "image_paths": paths,
                    "max_size": 800,
                }
                st.session_state["response"] = st.session_state["client"].run_graph(
                    input_data=input_data
                )
                if "user_edited_image_descriptions" not in st.session_state["response"]:
                    st.session_state["response"]["user_edited_image_descriptions"] = (
                        st.session_state["response"]["image_descriptions"]
                    )

                for resized_path, temp_path in zip(
                    st.session_state["response"]["resized_images"], paths
                ):
                    try:
                        logger.info(f"Removing temporary file: {temp_path}")
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to remove temporary file: {temp_path}")
                        logger.error(str(e))

                    st.session_state["images"].append(Image.open(resized_path).copy())
                    try:
                        logger.info(f"Removing resized image: {resized_path}")
                        os.remove(resized_path)
                    except Exception as e:
                        logger.error(f"Failed to remove resized image: {resized_path}")
                        logger.error(str(e))
            st.rerun()
    else:
        st.info("Please upload one or more images.")

else:
    st.success(
        f"Processed {len(st.session_state['response']['resized_images'])} image(s)"
    )
    image_descriptions = st.session_state["response"]["user_edited_image_descriptions"]
    for idx, (thumb, desc) in enumerate(
        zip(st.session_state["images"], image_descriptions)
    ):
        col1, col2 = st.columns([1, 2])
        thumb.thumbnail((256, 256))
        col1.image(thumb, caption=f"Image {idx + 1}", use_container_width=True)
        st.session_state["response"]["user_edited_image_descriptions"][idx] = (
            col2.text_area(
                label=f"Image {idx + 1} Description",
                value=desc,
                key=f"desc_{idx}",
                disabled=st.session_state["writing_in_progress"]
                or ("content" in st.session_state["response"]),
                height=250,
            )
        )

    # select persona
    select_persona = st.selectbox(
        "Select Persona",
        # List of personas come from persona directory
        [f[:-4] for f in os.listdir("persona") if f.endswith(".txt")],
        key="select_persona",
        disabled=st.session_state["writing_in_progress"]
        or ("content" in st.session_state["response"]),
    )
    text_area = st.text_area(
        "Optional: Additional Context",
        "",
        key="additional_text",
        disabled=st.session_state["writing_in_progress"]
        or ("content" in st.session_state["response"]),
    )

    if "content" not in st.session_state["response"]:
        if st.button(
            "Start Writing Process", disabled=st.session_state["writing_in_progress"]
        ):
            st.session_state["writing_in_progress"] = True
            st.session_state["additional_context"] = text_area
            st.session_state["persona_name"] = select_persona
            st.rerun()

    if (
        st.session_state["writing_in_progress"]
        and "content" not in st.session_state["response"]
    ):
        logger.info("Starting writing process...")
        input_data = {
            "persona_name": st.session_state["persona_name"],
            "additional_context": st.session_state["additional_context"],
            "user_edited_image_descriptions": st.session_state["response"][
                "user_edited_image_descriptions"
            ],
        }
        with st.spinner("Start writing process..."):
            with st.container(height=300):
                st.write_stream(
                    st.session_state["client"].run_graph_stream(
                        input_data=input_data, stream_mode="custom"
                    )
                )
        st.session_state["response"] = st.session_state["client"].get_state()
        st.session_state["writing_in_progress"] = False
        st.rerun()
    if (
        "content" in st.session_state["response"]
        and st.session_state["response"]["content"]
    ):
        with st.expander("Content"):
            st.markdown(st.session_state["response"]["content"])
        with st.expander("Caption"):
            st.markdown(st.session_state["response"]["caption"])

        # elif "localized_content" not in st.session_state["response"]:
        if not st.session_state["response"]["localized_content"]:
            language = st.selectbox(
                "Select Language",
                [
                    "Indonesian",
                    "Thailand",
                    "Japanese",
                    "Chinese",
                    "Korean",
                    "Spanish",
                    "French",
                ],
            )
            if st.button("Translate Content"):
                logger.info("Translating content...")
                input_data = {"target_language": language}
                logger.info(f"Input data: {input_data}")
                with st.spinner("Translating content..."):
                    with st.container(height=300):
                        st.write_stream(
                            st.session_state["client"].run_graph_stream(
                                input_data=input_data, stream_mode="custom"
                            )
                        )
                st.session_state["response"] = st.session_state["client"].get_state()
                st.rerun()

    if st.session_state["response"]["localized_content"]:
        for content in st.session_state["response"]["localized_content"]:
            st.markdown(content)
            st.markdown("---")
