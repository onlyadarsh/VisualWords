import os
import tempfile
import streamlit as st
from utils import extract_frames
from summarizer import summarize_captions

def run_video_mode(model, sample_rate):
    # ─── Inject CSS for a dashed drag-zone ───────────────────────────────────────
    st.markdown("""
    <style>
      /* style the file uploader container */
      div[data-testid="file-uploader"] > div {
          border: 2px dashed #ff6347 !important;
          border-radius: 10px !important;
          padding: 40px !important;
          text-align: center !important;
          background-color: #fff8f2 !important;
      }
      /* hide the default “Browse files” label */
      div[data-testid="file-uploader"] label {
          display: none !important;
      }
    </style>
    """, unsafe_allow_html=True)
    # ─────────────────────────────────────────────────────────────────────────────

    st.markdown("## 🎥 Drag & drop—or click—to upload a video")
    uploaded_video = st.file_uploader(
        "",  # label is hidden via CSS
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if not uploaded_video:
        return

    # save to temp file for OpenCV
    suffix = os.path.splitext(uploaded_video.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_video.read())
    tmp.close()
    video_path = tmp.name

    st.video(video_path)

    with st.spinner("Extracting frames..."):
        frames = extract_frames(
            video_path,
            fps=sample_rate,
            min_frames=3
        )

    for idx, frame in enumerate(frames, start=1):
        with st.spinner(f"Captioning frame {idx}/{len(frames)}..."):
            caps = model.predict_caption(frame)
        st.markdown(f"**Frame {idx}:**")
        st.image(frame, use_container_width=True)
        for j, cap in enumerate(caps, start=1):
            st.write(f"- Caption {j}: {cap}")

    # build summary from first caption of each frame
    all_captions = [model.predict_caption(frame)[0] for frame in frames]
    with st.spinner("Summarizing captions..."):
        summary = summarize_captions(all_captions)

    st.header("📜 Video Summary Caption:")
    st.success(summary)
