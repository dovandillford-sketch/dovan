import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import time

# ------------------- Styling -------------------
st.markdown("""
    <style>

    /* Style for buttons */
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
    }

    /* Footer styling */
    .custom-footer {
        text-align: center;
        width: 100%;
        padding: 10px;
        background-color: #8a2be2; /* blue violet */
        color: white;
        font-size: 14px;
        border-top: 1px solid #aaa;
        position: fixed;
        bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    model = YOLO("fire.pt")  # Ensure fire.pt is in the same folder
    return model

# ------------------- App Title -------------------
st.title("🔥AI-Based Early Fire Detection💨")

# ------------------- Load Model -------------------
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ------------------- Default Confidence -------------------
CONF_THRES = 0.25   # fixed threshold

# ------------------- Input Options -------------------
input_type = st.radio("Choose Input Source", ("Image", "Webcam"))

# ------------------- Image Detection -------------------
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Fire/Smoke in Image"):
            image = Image.open(uploaded_file)

            # Run detection
            results = model.predict(image, conf=CONF_THRES)

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated_img_array = results[0].plot()
                annotated_img = Image.fromarray(annotated_img_array[..., ::-1])

                st.subheader("Detection Results")
                st.image(annotated_img, caption="Detected Fire/Smoke", use_column_width=True)
                st.info(f"🔥 Detected **{len(results[0].boxes)}** instance(s).")

                # Download option
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    annotated_img.save(tmpfile.name, "JPEG")
                    with open(tmpfile.name, "rb") as file:
                        st.download_button(
                            "Download Annotated Image",
                            file,
                            "fire_detection.jpg",
                            "image/jpeg"
                        )
                    os.unlink(tmpfile.name)
            else:
                st.warning("⚠️ No fire or smoke detected.")

# ------------------- Webcam Detection -------------------
elif input_type == "Webcam":
    st.warning("⚠️ Press 'Start Webcam' to begin live detection")

    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    if not st.session_state.webcam_active:
        if st.button("Start Webcam"):
            st.session_state.webcam_active = True
            st.rerun()

    if st.session_state.webcam_active:
        FRAME_WINDOW = st.image([])
        stop_button = st.button("Stop Webcam")

        # Initialize webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check permissions and try again.")
            st.session_state.webcam_active = False
        else:
            st.success("✅ Webcam started successfully!")

            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to grab frame from webcam.")
                    time.sleep(0.1)
                    continue

                try:
                    results = model.predict(frame, conf=CONF_THRES, verbose=False)
                    annotated_frame = results[0].plot()
                    FRAME_WINDOW.image(annotated_frame, channels="BGR")
                except Exception as e:
                    st.error(f"❌ Error during detection: {e}")
                    break

                if stop_button:
                    st.session_state.webcam_active = False
                    st.rerun()

                time.sleep(0.05)

            cap.release()
            st.info("🛑 Webcam stopped")

# ------------------- Footer -------------------
st.markdown(
    """
    <div class="custom-footer">
        🔥 AI Fire & Smoke Detector | © 2025 Developed by <b>RUKUNDO Janvier</b>
    </div>
    """,
    unsafe_allow_html=True
)
