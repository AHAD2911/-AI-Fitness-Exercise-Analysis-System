import os
import streamlit as st
import cv2
import base64
import json
import time
import threading
import numpy as np
import tensorflow as tf
from video_manager import VideoManager
from analysis_engine import RealTimeAnalysisEngine
from rep_counters import SmartRepCounter

# --- BACKEND INITIALIZATION (Embedded) ---
@st.cache_resource
def initialize_engine():
    # Load model
    MODEL_PATH = "final_forthesis_bidirectionallstm_and_encoder_exercise_classif_model.h5"
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model loaded successfully")
        expected_frames, expected_features = model.input_shape[1], model.input_shape[2]
    except Exception as e:
        print(f"[INFO] Model not found, using dummy: {e}")
        expected_frames, expected_features = 10, 66
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(expected_frames, expected_features)),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')

    video_manager = VideoManager()
    smart_rep_counter = SmartRepCounter()
    return model, video_manager, smart_rep_counter

model, video_manager, smart_rep_counter = initialize_engine()

st.set_page_config(page_title="Pro AI Fitness Coach", layout="wide")

# --- THREADED CAMERA CLASS ---
class VideoStream:
    """Grabs frames in a background thread to prevent UI freezing."""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        # Set capture resolution (configurable)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Eliminate lag/buffer
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- UI SIDEBAR ---
st.sidebar.title("🏋️ Control Panel")
exercise_name = st.sidebar.text_input("Exercise ID", value="bicep_curls").lower()
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Reference", type=['mp4', 'mov'])

# Throttle controls to reduce camera/backend load
send_every_n = st.sidebar.slider("Send every Nth frame", 1, 10, 3)
capture_width = st.sidebar.selectbox("Capture width", [320, 480, 640], index=0)
capture_height = int(capture_width * 3 / 4)
jpeg_quality = st.sidebar.slider("JPEG quality (lower=smaller)", 30, 95, 60)

if st.sidebar.button("Register & Analyze Reference"):
    if uploaded_file:
        with st.spinner("Analyzing reference..."):
            try:
                # Save and analyze directly
                video_info = video_manager.save_video(exercise_name, uploaded_file, uploaded_file.name)
                analysis_result = video_manager.analyze_reference_video(video_info["id"], exercise_name)
                st.sidebar.success(f"Reference Registered! Estimated Reps: {analysis_result.total_reps}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# --- MAIN DASHBOARD ---
st.title("AI Exercise Analysis")
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()
    status_text = st.empty()

with col2:
    st.markdown("### Metrics")
    rep_count = st.empty()
    stage_label = st.empty()
    angle_gauge = st.empty()
    feedback_note = st.empty()
    sim_score = st.empty()
    per_joint_area = st.empty()

# --- ANALYSIS LOOP ---
def start_workout():
    session_id = f"user_{int(time.time())}"
    
    # Initialize Engine for this session
    engine = RealTimeAnalysisEngine(
        model=model,
        labels=[],
        video_manager=video_manager,
        session_id=session_id,
        target_exercise=exercise_name
    )
    
    # Initialize Threaded Camera
    vs = VideoStream(src=0, width=capture_width, height=capture_height).start()
    
    status_text.success(f"AI Engine Ready for {exercise_name}")
    
    stop_button = st.button("🛑 Stop Workout")
    
    frame_count = 0
    while not stop_button:
        frame = vs.read()
        if frame is None: continue
        
        frame_count += 1
        
        # Throttle analysis
        if frame_count % send_every_n == 0:
            # Encode to JPEG bytes (as the engine expects image_data bytes)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            image_bytes = buffer.tobytes()

            # Analyze frame directly
            result = engine.analyze_frame(image_bytes)

            if result:
                # Update Metrics
                rep_count.metric("REPS", result.rep_count)
                stage_label.info(f"STAGE: {result.stage.upper()}")
                angle_gauge.metric("ANGLE", f"{round(result.current_angle, 1)}°")
                feedback_note.warning(result.form_feedback)
                sim_score.progress(min(1.0, max(0.0, result.similarity_score))) 

                # Per-joint details
                if result.per_joint:
                    lines = []
                    for jname, jv in result.per_joint.items():
                        reps = jv.get("reps", 0)
                        ang = round(float(jv.get("angle", 0.0)), 1)
                        sim = round(float(jv.get("similarity", 0.0)), 2)
                        stage = jv.get("stage", "")
                        lines.append(f"**{jname.replace('_',' ').title()}** — Reps: {reps} | Angle: {ang}° | Sim: {sim} | Stage: {stage}")
                    per_joint_area.markdown("\n\n".join(lines))
                else:
                    per_joint_area.markdown("")

                # Draw landmarks on the preview frame
                if result.pose_landmarks:
                    for lm in result.pose_landmarks:
                        h, w, _ = frame.shape
                        cv2.circle(frame, (int(lm['x']*w), int(lm['y']*h)), 4, (0, 255, 0), -1)

        # Update Video Display
        frame_placeholder.image(frame, channels="BGR")
        
        # Minor sleep to avoid CPU pegging
        time.sleep(0.01)
        
    vs.stop()
    status_text.info("Workout Stopped")

# --- START BUTTON ---
if st.button("🚀 Start Live Workout"):
    start_workout()
