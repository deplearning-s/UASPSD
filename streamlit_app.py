import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image

# =========================
# Load YOLO model (Cloud-safe)
# =========================
@st.cache_resource
def load_model():
    return YOLO("best_helmet_yolov8.pt")  # pastikan file ini ada di repo

model = load_model()

# =========================
# UI
# =========================
st.title("Deteksi Kepatuhan Helm Pada Pengendara Sepeda Motor")
st.write("Upload gambar atau video untuk mendeteksi penggunaan helm.")

uploaded_file = st.file_uploader(
    "Upload image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# =========================
# Image Processing
# =========================
if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        results = model(img_array)
        annotated_img = results[0].plot()

        st.image(
            annotated_img,
            caption="Hasil Deteksi",
            use_column_width=True
        )

    # =========================
    # Video Processing
    # =========================
    elif uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame, use_column_width=True)

        cap.release()
