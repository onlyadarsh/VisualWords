import streamlit as st

def render_sidebar():
    st.sidebar.markdown("""
        <h1 style="font-family: 'Arial'; color: #ff6347; text-align: center;">🖼️🧑‍💻 Visual Words</h1>
        <p style="font-size: 16px; text-align: center;">A visual storytelling tool for generating captions from images and videos.</p>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    # Mode selector (unique key)
    mode = st.sidebar.radio(
        "Select mode:",
        ["Image📷", "Video📽️"],
        key="mode_selector"
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # How many captions per item
    top_k = st.sidebar.slider(
        "Captions per item",
        min_value=1,
        max_value=5,
        value=3,
        key="top_k_slider"
    )

    # If video, show FPS slider and force large model
    if mode == "Video📽️":
        sample_rate = st.sidebar.slider(
            "Frames per second",
            min_value=1,
            max_value=2,
            value=1,
            key="fps_selector"
        )
        model_name = "Salesforce/blip-image-captioning-large"
    else:
        sample_rate = None
        model_name = st.sidebar.selectbox(
            "Model variant",
            [
                "Salesforce/blip-image-captioning-base",
                "Salesforce/blip-image-captioning-large"
            ],
            key="model_selector"
        )

    st.sidebar.markdown("""
        <br><br>
        <p style="font-size: 14px; text-align: center; color: gray;">
            Powered by <a href="https://huggingface.co" target="_blank">Hugging Face</a>
        </p>
    """, unsafe_allow_html=True)

    return mode, top_k, sample_rate, model_name
