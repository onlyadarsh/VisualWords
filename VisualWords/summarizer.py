# summarizer.py

from transformers import pipeline

# load once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_captions(captions: list[str]) -> str:
    text = " ".join(captions)
    # summarize
    summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
    return summary[0]['summary_text']
