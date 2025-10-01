import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile

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
    model = YOLO("fire.pt")  # Ensure fire.pt is in the same folder
    return model

st.title("🔥 AI-Based Fire Detection 💨")

# ------------------- Load Model -------------------
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ------------------- Confidence Slider -------------------
conf_thres = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

# ------------------- Sidebar Input Options -------------------
input_type = st.radio("Choose Input Source", ("Image", "Webcam"))

# ------------------- Image Upload Detection -------------------
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Fire/Smoke in Image"):
            image = Image.open(uploaded_file)

            # Run detection
            results = model.predict(image, conf=conf_thres)

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated_img_array = results[0].plot()
                annotated_img = Image.fromarray(annotated_img_array[..., ::-1])

                st.subheader("Detection Results")
                st.image(annotated_img, caption="Detected Fire/Smoke", use_column_width=True)
                st.info(f"🔥 Detected **{len(results[0].boxes)}** instance(s).")

                # Option to download result
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    annotated_img.save(tmpfile.name)
                    st.download_button("Download Annotated Image", open(tmpfile.name, "rb"), "fire_detection.jpg")

            else:
                st.warning("⚠️ No fire or smoke detected.")

# ------------------- Webcam Detection -------------------
elif input_type == "Webcam":
    st.warning("⚠️ Press 'Start Webcam' to begin live detection")

    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)  # Open default webcam

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access webcam.")
                break

            # Run YOLO detection
            results = model.predict(frame, conf=conf_thres)

            annotated_frame = results[0].plot()

            FRAME_WINDOW.image(annotated_frame, channels="BGR")

        cap.release()

