import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007BFF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    label[data-testid="stMarkdownContainer"] {
        color: green;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .green-upload label {
        color: green;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model once for performance
@st.cache_resource
def load_model():
    model = YOLO("fire.pt")  # Make sure 'fire.pt' is in your project folder
    return model

model = None

st.title("🔥💨AI-Based Fire Detection App💨🔥")

# Load model safely
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Option to choose input type
input_type = st.radio("Select input type:", ("Image", "Video"))

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Fire/Smoke the in Image"):
            image = Image.open(uploaded_file)
            results = model(image)
            
            if results and results[0].boxes:
                annotated_img_array = results[0].plot()
                annotated_img = Image.fromarray(annotated_img_array[..., ::-1])
                
                st.subheader("Detection Results")
                st.image(annotated_img, caption="Detected Fire", use_column_width=True)
                st.info(f"Detected **{len(results[0].boxes)}** instances of fire.")
            else:
                st.warning("No detections found.")

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
        max_frames = 100  # limit frames for demo, remove or increase for full video
        
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
            
            cap.release()
            out.release()
            
            video_file = open(output_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            os.remove(tfile.name)

