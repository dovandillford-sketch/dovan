import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# ------------------- Styling -------------------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    model = YOLO("fire.pt")  # Ensure fire.pt is in your project folder
    return model

st.title("🔥 AI-Based Fire Detection App 💨")

# ------------------- Load Model Safely -------------------
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------- Input Type Selection -------------------
input_type = st.radio("Select input type:", ("Image", "Video", "Webcam"))

# ------------------- Image Upload -------------------
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Fire/Smoke in Image"):
            image = Image.open(uploaded_file)
            results = model(image)

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated_img_array = results[0].plot()
                annotated_img = Image.fromarray(annotated_img_array[..., ::-1])

                st.subheader("Detection Results")
                st.image(annotated_img, caption="Detected Fire", use_column_width=True)
                st.info(f"Detected **{len(results[0].boxes)}** instance(s) of fire.")
            else:
                st.warning("No fire or smoke detected.")

# ------------------- Video Upload -------------------
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video file...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "output.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        frame_count = 0
        max_frames = 100  # You can change or remove this limit
        progress_bar = st.progress(0)

        if st.button("Detect Fire/Smoke in Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                results = model(frame[..., ::-1])
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                stframe.image(annotated_frame[..., ::-1], channels="RGB")

                frame_count += 1
                progress_bar.progress(min(frame_count / max_frames, 1.0))

            cap.release()
            out.release()

            video_file = open(output_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.download_button("Download Processed Video", video_bytes, file_name="detected_fire.mp4")

            os.remove(tfile.name)
            os.remove(output_path)

# ------------------- Webcam Live Detection -------------------
elif input_type == "Webcam":
    if st.button("Start Webcam Fire Detection"):
        cap = cv2.VideoCapture(0)  # Default webcam
        stframe = st.empty()
        st.info("Press 'Stop' to end webcam detection.")

        stop_button = st.button("Stop")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            results = model(frame[..., ::-1])
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame[..., ::-1], channels="RGB")

        cap.release()
