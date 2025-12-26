import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Helm", layout="wide")

# Load YOLO model (path RELATIF)
@st.cache_resource
def load_model():
    return YOLO("best_helmet_yolov8.pt")

model = load_model()

st.title("🪖 Deteksi Kepatuhan Helm Pada Pengendara Sepeda Motor")
st.write("Upload gambar atau video untuk mendeteksi penggunaan helm.")

uploaded_file = st.file_uploader(
    "Upload image atau video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # ================= IMAGE =================
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model(img)
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Hasil Deteksi", use_container_width=True)

    # ================= VIDEO =================
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            output_frame = results[0].plot()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            stframe.image(output_frame, use_container_width=True)

        cap.release()
