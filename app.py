import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import base64

# ------------------- Styling -------------------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
    }
    label[data-testid="stMarkdownContainer"], .green-upload label {
        color: green !important;
        font-weight: bold !important;
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
input_type = st.radio("Select input type:", ("Image", "Video"))

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
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile_path = tfile.name

        cap = cv2.VideoCapture(tfile_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        frame_count = 0
        max_frames = 100  # Limit for performance

        progress_bar = st.progress(0)

        if st.button("Detect Fire/Smoke in Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                rgb_frame = frame[..., ::-1]  # Convert BGR to RGB
                results = model(rgb_frame)
                annotated_frame = results[0].plot()  # RGB image
                annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)

                stframe.image(annotated_frame, channels="RGB")
                frame_count += 1
                progress_bar.progress(min(frame_count / max_frames, 1.0))

            cap.release()
            out.release()

            # ------------------- Embed Video with Base64 -------------------
            def embed_video_base64(video_path):
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                    b64 = base64.b64encode(video_data).decode()
                    video_html = f"""
                    <video width="100%" controls autoplay muted loop>
                        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """
                    st.markdown(video_html, unsafe_allow_html=True)

            st.subheader("Processed Video")
            embed_video_base64(output_path)

            # Allow user to download video
            with open(output_path, 'rb') as f:
                st.download_button("Download Processed Video", f.read(), file_name="detected_fire.mp4")

            # ------------------- Clean up temp files -------------------
            os.remove(tfile_path)
            os.remove(output_path)

