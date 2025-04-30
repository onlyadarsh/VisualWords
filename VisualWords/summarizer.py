# summarizer.py

import os
from transformers import pipeline

# Load Hugging Face token from environment or Streamlit secrets
hf_token = os.environ.get("HF_TOKEN")  # or st.secrets["HF_TOKEN"] if using Streamlit secrets

# Load summarizer with authentication
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    use_auth_token=hf_token
)
def summarize_captions(captions: list[str]) -> str:
    text = " ".join(captions)
    summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
    return summary[0]['summary_text']
