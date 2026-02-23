# Module Quick Reference

## File Organization

```
AI fitnees project/
├── Main.py                    # Entry point (112 lines) ⭐ START HERE
├── models.py                  # Data structures (27 lines)
├── trajectory.py              # Movement tracking (56 lines)
├── rep_counters.py            # Rep counting logic (433 lines)
├── video_manager.py           # Video handling (164 lines)
├── connection_manager.py      # WebSocket management (47 lines)
├── analysis_engine.py         # Frame analysis (253 lines)
├── api_routes.py              # API endpoints (169 lines)
├── Main.py.backup             # Original Main.py (1634 lines)
├── trainer.py                 # Alternate trainer implementation
├── frontend.py                # Streamlit UI
├── requirements.txt           # Dependencies
└── REFACTORING_SUMMARY.md     # This refactoring document
```

## Module Descriptions

### `models.py`
**Purpose**: Data structures only
**Classes**:
- `ExerciseAnalysis` - Frame analysis result
- `VideoAnalysisResult` - Reference video analysis
**Size**: 27 lines | **Dependencies**: None

### `trajectory.py`
**Purpose**: Track movement trajectories
**Classes**:
- `TrajectoryTracker` - Stores angles, timestamps, velocities, rep boundaries
**Key Methods**:
- `add_point(angle, timestamp)` - Add data point
- `get_current_trajectory()` - Return trajectory dict
- `mark_rep_completion()` - Mark rep done
- `get_recent_range()` - Get ROM in window
**Size**: 56 lines | **Dependencies**: None

### `rep_counters.py`
**Purpose**: All rep counting and form analysis
**Classes**:
- `ExerciseCounter` - Base class with angle calculation
- `ImprovedGenericCounter` - 3-stage counter (up/down/neutral)
- `SixPointRepCounter` - 6-stage counter
- `SmartRepCounter` - Router delegating to appropriate counter
**Key Methods**:
- `analyze_landmarks(landmarks)` → `(rep_count, angle, similarity)`
- `register_exercise_from_rule(exercise, rule)` - Register with thresholds
- `get_debug_info()` - Get counter state
**Size**: 433 lines | **Dependencies**: `trajectory.py`, mediapipe

### `video_manager.py`
**Purpose**: Manage video uploads and extract exercise patterns
**Classes**:
- `VideoManager` - Save videos, analyze references, extract rep rules
**Key Methods**:
- `save_video(exercise_type, video_file)` - Upload and store
- `analyze_reference_video(video_id, exercise_type)` - Auto-detect joint angles
- `get_reference_video(exercise_type)` - Retrieve analysis
**Features**:
- Auto-detects best moving joint (highest angle variance)
- Extracts angle percentiles (up/down thresholds)
- Estimates rep count from angle pattern
**Size**: 164 lines | **Dependencies**: `models.py`, `rep_counters.py`, mediapipe, cv2

### `connection_manager.py`
**Purpose**: WebSocket connection lifecycle
**Classes**:
- `ConnectionManager` - Manage active connections and analysis engines
**Key Methods**:
- `async connect(websocket, exercise_type, ...)` - New connection
- `disconnect(session_id)` - Clean up
- `get_analysis_engine(session_id)` - Get engine for session
- `async send_json(data, session_id)` - Send to client
**Size**: 47 lines | **Dependencies**: `analysis_engine.py`

### `analysis_engine.py`
**Purpose**: Real-time frame processing
**Classes**:
- `RealTimeAnalysisEngine` - Analyze frames, generate feedback
**Key Methods**:
- `analyze_frame(image_data)` → `ExerciseAnalysis` - Main processing
- `reset_session()` - Reset counter
- `get_session_stats()` - Debug info
**Features**:
- Decodes base64 frames
- Runs MediaPipe pose detection
- Maintains pose buffer
- Generates form feedback
- Extracts key landmarks for visualization
**Size**: 253 lines | **Dependencies**: `models.py`, `rep_counters.py`, mediapipe, cv2

### `api_routes.py`
**Purpose**: FastAPI endpoint definitions
**Functions**:
- `register_routes(app, connection_manager, video_manager, smart_rep_counter)` - Register all routes
**Endpoints**:
- `WebSocket /ws/live-analysis/{session_id}?exercise_type={ex}` - Live analysis
- `POST /upload-reference-video?exercise_type={ex}` - Upload reference
- `POST /analyze-frame?exercise_type={ex}` - Single frame analysis
- `GET /` - Health check
**Size**: 169 lines | **Dependencies**: All above modules

### `Main.py` (NEW)
**Purpose**: Application entry point and initialization
**Responsibilities**:
1. Load TensorFlow model
2. Initialize FastAPI app
3. Create global managers
4. Register routes
5. Start uvicorn server
**Usage**:
```bash
python Main.py
```
**Size**: 112 lines | **Dependencies**: All above modules

## How They Connect

```
User Request
    ↓
FastAPI (api_routes.register_routes)
    ├─→ WebSocket: connection_manager.connect()
    │       ↓
    │   RealTimeAnalysisEngine
    │       ├─→ video_manager.get_reference_video()
    │       └─→ SmartRepCounter.analyze_frame()
    │           └─→ ImprovedGenericCounter.analyze_landmarks()
    │               ├─→ TrajectoryTracker.add_point()
    │               └─→ returns (rep_count, angle, similarity)
    │
    ├─→ Upload: video_manager.save_video()
    │   └─→ video_manager.analyze_reference_video()
    │       └─→ SmartRepCounter.register_exercise_from_rule()
    │
    └─→ Analyze: SmartRepCounter.analyze_frame()
```

## Import Examples

```python
# Example 1: Test rep counter offline
from rep_counters import SmartRepCounter
from trajectory import TrajectoryTracker

counter = SmartRepCounter()
counter.register_exercise_from_rule("bicep_curls", {
    "indices": (11, 13, 15),
    "up_threshold": 160,
    "down_threshold": 50
})
reps, angle, sim = counter.analyze_frame("bicep_curls", landmarks)
```

```python
# Example 2: Track trajectory manually
from trajectory import TrajectoryTracker

tracker = TrajectoryTracker()
for angle in angles:
    tracker.add_point(angle, time.time())
    if rep_completed:
        tracker.mark_rep_completion()

data = tracker.get_current_trajectory()
print(f"Angles: {data['angles']}")
print(f"Reps at indices: {data['rep_boundaries']}")
```

```python
# Example 3: Analyze video reference
from video_manager import VideoManager
from fastapi import UploadFile

vm = VideoManager()
video_info = vm.save_video("push_ups", uploaded_file)
result = vm.analyze_reference_video(video_info["id"], "push_ups")
print(f"Estimated reps: {result.total_reps}")
print(f"Up threshold: {result.rep_rule['up_threshold']:.1f}°")
```

## Running with Virtual Environment

```bash
# Activate venv (Windows)
.\.venv\Scripts\Activate.ps1

# Run server
python Main.py

# Test imports
python -c "from rep_counters import SmartRepCounter; print('OK')"

# Or use venv python directly (without activation)
.\.venv\Scripts\python.exe Main.py
```

## Line Count Breakdown

| File | Lines | Purpose |
|------|-------|---------|
| models.py | 27 | Data structures |
| trajectory.py | 56 | Movement tracking |
| rep_counters.py | 433 | Rep counting |
| video_manager.py | 164 | Video handling |
| connection_manager.py | 47 | WebSocket mgmt |
| analysis_engine.py | 253 | Frame analysis |
| api_routes.py | 169 | API endpoints |
| Main.py | 112 | Entry point |
| **TOTAL** | **1,261** | **Working code** |
| Main.py.backup | 1,634 | Original monolith |
| **Reduction** | -373 lines | **23% smaller** |

*Note: Code reduction due to removed duplication, simplified structure, and consolidated imports.*

## Testing

```bash
# Test all module imports
.\.venv\Scripts\python.exe -c "
from trajectory import TrajectoryTracker
from rep_counters import SmartRepCounter
from models import ExerciseAnalysis
from video_manager import VideoManager
from connection_manager import ConnectionManager
from analysis_engine import RealTimeAnalysisEngine
print('[SUCCESS] All modules imported')
"

# Test Main.py
.\.venv\Scripts\python.exe -c "import Main; print('[SUCCESS] Main.py loads')"
```

## Migration Checklist

- [x] Extract models.py
- [x] Extract trajectory.py
- [x] Extract rep_counters.py
- [x] Extract video_manager.py
- [x] Extract connection_manager.py
- [x] Extract analysis_engine.py
- [x] Extract api_routes.py
- [x] Update Main.py
- [x] Fix mediapipe imports
- [x] Test all imports with .venv
- [x] Backup original Main.py
- [x] Create documentation
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Deploy and monitor
