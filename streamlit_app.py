import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Deteksi Helm", layout="wide")

# Load YOLO model (RELATIVE PATH)
@st.cache_resource
def load_model():
    return YOLO("best_helmet_yolov8.pt")

model = load_model()

st.title("🪖 Deteksi Kepatuhan Helm Pengendara Motor")
st.write("Upload **gambar atau video** untuk mendeteksi penggunaan helm.")

uploaded_file = st.file_uploader(
    "Upload file",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    ### ================= IMAGE ================= ###
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(img)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Hasil Deteksi", use_column_width=True)

    ### ================= VIDEO ================= ###
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        fps_limit = 5  # batasi FPS biar cloud tidak crash
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % fps_limit != 0:
                continue

            results = model(frame)
            output = results[0].plot()
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            stframe.image(output_rgb, use_column_width=True)

        cap.release()
        os.unlink(tfile.name)

