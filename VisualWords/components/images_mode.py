import streamlit as st
from PIL import Image

def run_image_mode(model):
    # —————— Inject custom CSS now ——————
    st.markdown("""
    <style>
  /* Catch multiple internal class/ID patterns Streamlit might use */
  div[data-testid="file-uploader"] > div,
  div[data-testid="stFileUploader"] > div,
  div[role="button"] {
      border: 2px dashed #ff6347 !important;
      border-radius: 10px !important;
      padding: 40px !important;
      text-align: center !important;
      background-color: #fff8f2 !important;
  }
  /* Hide the native label/text inside those containers */
  div[data-testid="file-uploader"] label,
  div[data-testid="stFileUploader"] label,
  div[role="button"] label {
      display: none !important;
  }
</style>

    """, unsafe_allow_html=True)
    # ——————————————————————————————

    st.markdown("## 📷 Drag & drop—or click—to upload an image")
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"], 
        key="image_uploader"
    )
    if not uploaded_file:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    with st.spinner("Generating captions…"):
        captions = model.predict_caption(image)

    for i, cap in enumerate(captions, start=1):
        st.write(f"**Caption {i}:** {cap}")

