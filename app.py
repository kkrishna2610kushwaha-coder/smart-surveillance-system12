import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart Surveillance System", page_icon="📹", layout="wide")

st.title("📹 Smart Surveillance System")
st.caption("Employee Monitoring and Indexing using Deep Learning")
st.info(
    "This educational demo detects people in uploaded images and videos using YOLO11. "
    "It does not identify a person's name or verify employee identity."
)

@st.cache_resource
def load_model():
    # Ultralytics downloads this small pretrained model automatically on first run.
    return YOLO("yolo11n.pt")

@st.cache_data
def empty_records():
    return pd.DataFrame(columns=["Date", "Time", "People_Count", "Source", "File_Name"])

def add_record(count, source, filename):
    now = datetime.now()
    row = {
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "People_Count": int(count),
        "Source": source,
        "File_Name": filename,
    }
    if "records" not in st.session_state:
        st.session_state.records = empty_records()
    st.session_state.records = pd.concat(
        [st.session_state.records, pd.DataFrame([row])], ignore_index=True
    )

def detect_people(model, image_array, confidence):
    # COCO class 0 is "person".
    results = model.predict(
        source=image_array, classes=[0], conf=confidence, verbose=False
    )
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    count = 0 if result.boxes is None else len(result.boxes)
    return annotated_rgb, count

if "records" not in st.session_state:
    st.session_state.records = empty_records()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.40, 0.05)
    st.caption("Higher confidence may reduce false detections.")
    module = st.radio(
        "Choose Module",
        ["Home", "Image Detection", "Video Monitoring", "Employee Records"],
    )

try:
    with st.spinner("Loading YOLO11 deep-learning model..."):
        model = load_model()
    st.sidebar.success("AI Model Loaded")
except Exception as exc:
    st.error("The AI model could not be loaded. Check the app logs and dependencies.")
    st.exception(exc)
    st.stop()

if module == "Home":
    st.subheader("Project Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("AI Model", "YOLO11n")
    c2.metric("Technology", "Deep Learning")
    c3.metric("System", "Ready")
    st.markdown(
        """
        **Main features**
        - Detect people in uploaded CCTV-style images.
        - Analyze uploaded video and estimate the number of people in sampled frames.
        - Store detection summaries in an in-session CSV report.
        - Download the report for project documentation.

        **How it works:** YOLO11 is a pretrained object-detection neural network.
        The app filters its predictions to the `person` class (COCO class 0).
        """
    )

elif module == "Image Detection":
    st.subheader("Image-Based Person Detection")
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="image_upload"
    )
    if uploaded_image:
        pil_image = Image.open(uploaded_image).convert("RGB")
        image_array = np.array(pil_image)
        with st.spinner("Detecting people..."):
            annotated, count = detect_people(model, image_array, confidence)
        left, right = st.columns(2)
        left.image(image_array, caption="Original Image", use_container_width=True)
        right.image(annotated, caption="Detection Result", use_container_width=True)
        st.success(f"People detected: {count}")
        if st.button("Save image result to records"):
            add_record(count, "Image", uploaded_image.name)
            st.success("Detection summary saved.")

elif module == "Video Monitoring":
    st.subheader("Video-Based Person Detection")
    uploaded_video = st.file_uploader(
        "Upload a CCTV video", type=["mp4", "avi", "mov"], key="video_upload"
    )
    frame_step = st.select_slider(
        "Analyze every Nth frame (larger value = faster processing)",
        options=[1, 2, 3, 5, 8, 10],
        value=5,
    )
    if uploaded_video and st.button("Start video analysis"):
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        temp_path = None
        cap = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_video.getbuffer())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Could not open this video. Try an MP4 file encoded with H.264.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()
                sampled_counts = []
                frame_index = 0
                processed = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_index % frame_step == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        annotated, count = detect_people(model, rgb, confidence)
                        sampled_counts.append(count)
                        processed += 1
                        preview.image(annotated, caption=f"Analyzed frame {frame_index}", use_container_width=True)
                        status.write(f"Processed {processed} sampled frames")
                        if total_frames > 0:
                            progress.progress(min(frame_index / total_frames, 1.0))
                    frame_index += 1

                progress.progress(1.0)
                cap.release()
                cap = None

                if sampled_counts:
                    max_people = max(sampled_counts)
                    avg_people = sum(sampled_counts) / len(sampled_counts)
                    st.success("Video analysis completed.")
                    a, b, c = st.columns(3)
                    a.metric("Frames analyzed", processed)
                    b.metric("Maximum people in a frame", max_people)
                    c.metric("Average people per sampled frame", f"{avg_people:.1f}")
                    if st.button("Save video summary to records"):
                        add_record(max_people, "Video (maximum per sampled frame)", uploaded_video.name)
                        st.success("Video summary saved.")
                else:
                    st.warning("No frames could be analyzed.")
        except Exception as exc:
            st.error("Video processing failed. Try a shorter MP4 video.")
            st.exception(exc)
        finally:
            if cap is not None:
                cap.release()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

elif module == "Employee Records":
    st.subheader("Detection Records")
    records = st.session_state.records
    if records.empty:
        st.warning("No records yet. Run image or video detection and save a result.")
    else:
        st.dataframe(records, use_container_width=True)
        csv_data = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download surveillance_report.csv",
            data=csv_data,
            file_name="surveillance_report.csv",
            mime="text/csv",
        )
        if st.button("Clear records"):
            st.session_state.records = empty_records()
            st.rerun()

st.divider()
st.caption("Built with Python • Streamlit • YOLO11 • OpenCV • Deep Learning")
