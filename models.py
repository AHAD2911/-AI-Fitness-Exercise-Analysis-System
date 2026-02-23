"""
Data models and structures for exercise analysis.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ExerciseAnalysis:
    """Represents a single frame analysis result."""
    exercise_name: str
    confidence: float
    rep_count: int
    current_angle: float
    form_feedback: str
    similarity_score: float
    stage: str
    timestamp: float
    is_matching_reference: bool
    pose_landmarks: Optional[List[Dict]] = None
    # Per-joint live results: {joint_name: {"reps":int, "angle":float, "similarity":float}}
    per_joint: Optional[Dict[str, Dict[str, Any]]] = None
    stuck_joints: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """Results from analyzing a reference video."""
    total_reps: int
    average_form_score: float
    exercise_detected: str
    key_poses: List[List[float]]
    duration: float
    analysis_frames: int
    rep_rule: Dict[str, Any] = None
    # Per-joint angle traces: {joint_name: [angles...]}
    angle_traces_by_joint: Optional[Dict[str, List[float]]] = None
    # Per-joint frame-to-frame deltas: {joint_name: [deltas...]}
    angle_deltas_by_joint: Optional[Dict[str, List[float]]] = None
    # Cleaned traces after ignoring small fluctuations (±10°)
    cleaned_traces_by_joint: Optional[Dict[str, List[float]]] = None
    # Per-joint thresholds: {joint_name: {'min':..,'mid':..,'max':..}}
    joint_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    # Per-joint similarity / movement score (0..1)
    joint_similarity: Optional[Dict[str, float]] = None
    # List of rep rules discovered for multiple joints (each a dict with indices/up/down/joint_name)
    rep_rules: Optional[List[Dict[str, Any]]] = None
    # Estimated reps per joint discovered in reference analysis
    per_joint_estimates: Optional[Dict[str, int]] = None
