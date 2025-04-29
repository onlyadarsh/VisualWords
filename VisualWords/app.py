import os
import tempfile
import streamlit as st
from PIL import Image
from model import ImageCaptionGenerator
from utils import extract_frames
from summarizer import summarize_captions
from PIL import Image
from components.sidebar import render_sidebar
from components.images_mode import run_image_mode
from components.Video_mode import run_video_mode
from pathlib import Path
import torch

st.set_page_config(page_title="Visual Words", layout="wide")
st.title(" Visual Words")


mode, top_k, sample_rate, model_name = render_sidebar()

@st.cache_resource
def load_model(k, model_name):
    # always use fp16 + compile for best performance
    return ImageCaptionGenerator(
        top_k=k,
        use_fp16=True,
        use_compile=True,
        model_name=model_name
    )
    
model = load_model(k=top_k, model_name=model_name)

if mode == "Image📷":
    run_image_mode(model)
else:
    run_video_mode(model, sample_rate)
