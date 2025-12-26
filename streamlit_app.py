import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Helm", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("best_helmet_yolov8.pt")

model = load_model()

st.title("🪖 Deteksi Kepatuhan Helm (Image Only)")
st.write("Upload GAMBAR. Video TIDAK didukung di Streamlit Cloud.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)
    annotated = results[0].plot()

    st.image(
        annotated,
        caption="Hasil Deteksi",
        use_container_width=True
    )
