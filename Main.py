"""
═══════════════════════════════════════════════════════════════════════════════
    AI FITNESS EXERCISE ANALYSIS SYSTEM - BACKEND (Main.py - REFACTORED)
    
    PURPOSE: Real-time exercise analysis with pose detection and rep counting
    
    MODULAR COMPONENTS:
    1. models.py - Data structures (ExerciseAnalysis, VideoAnalysisResult)
    2. trajectory.py - Trajectory tracking for movement analysis
    3. rep_counters.py - Rep counting classes (Improved, SixPoint, Smart)
    4. video_manager.py - Video upload & reference extraction
    5. connection_manager.py - WebSocket connection management
    6. analysis_engine.py - Real-time frame analysis engine
    7. api_routes.py - FastAPI endpoints (WebSocket, Upload, Analyze)
    
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import uvicorn

# Import modular components
from models import ExerciseAnalysis, VideoAnalysisResult
from trajectory import TrajectoryTracker
from rep_counters import (
    ExerciseCounter, 
    ImprovedGenericCounter, 
    SixPointRepCounter, 
    SmartRepCounter
)
from video_manager import VideoManager
from connection_manager import ConnectionManager
from analysis_engine import RealTimeAnalysisEngine
from api_routes import register_routes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: APPLICATION SETUP & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize FastAPI app with CORS middleware
app = FastAPI(title="AI Exercise Analysis System with WebSocket Streaming")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL LOADING & MEDIAPOSE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Load TensorFlow Keras model for exercise classification
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "final_forthesis_bidirectionallstm_and_encoder_exercise_classif_model.h5",
)

# Initialize MediaPipe Pose
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Load model or create compatible dummy model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully")
    
    try:
        expected_frames, expected_features = model.input_shape[1], model.input_shape[2]
        print(f"[INFO] Model expects input shape: (batch, {expected_frames}, {expected_features})")
    except:
        expected_frames, expected_features = 10, 66
        print("[INFO] Could not determine model input shape, using defaults")
        
except Exception as e:
    print(f"[INFO] Model not found, using dummy model: {e}")
    expected_frames, expected_features = 10, 66  # 33 landmarks * 2 (x,y)
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(expected_frames, expected_features)),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print("[INFO] Created dummy model with compatible input shape")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: INITIALIZE GLOBAL MANAGERS & COUNTERS
# ═══════════════════════════════════════════════════════════════════════════════

LABELS = []
smart_rep_counter = SmartRepCounter()
connection_manager = ConnectionManager()
video_manager = VideoManager()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: REGISTER API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

register_routes(app, connection_manager, video_manager, smart_rep_counter)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN APPLICATION ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" AI FITNESS EXERCISE ANALYSIS SYSTEM - BACKEND SERVER")
    print("="*80)
    print(f"\n[INFO] Starting server on http://0.0.0.0:8000")
    print(f"[INFO] Currently registered exercises: {list(video_manager.supported_exercises)}")
    print(f"[INFO] API documentation at: http://localhost:8000/docs")
    print(f"[INFO] WebSocket endpoint: ws://localhost:8000/ws/live-analysis/{{session_id}}?exercise_type={{exercise}}")
    print(f"[INFO] Video upload endpoint: POST /upload-reference-video")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=300,
        ws_ping_interval=30,
        ws_ping_timeout=10
    )
