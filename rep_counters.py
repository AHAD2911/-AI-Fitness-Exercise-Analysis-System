"""
Exercise counter classes for rep counting and form analysis.
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, List
from collections import deque
import mediapipe as mp

from trajectory import TrajectoryTracker

mp_pose = mp.solutions.pose


class ExerciseCounter:
    """Base exercise counter class."""
    def __init__(self):
        self.count = 0
        self.stage = "neutral"
        self.last_angle = 0.0
        self.angle_history = []
        self.form_scores = []
        
    def reset(self):
        self.count = 0
        self.stage = "neutral"
        self.last_angle = 0.0
        self.angle_history = []
        self.form_scores = []
    
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate joint angle from three landmarks."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_form_score(self, angle: float, landmarks) -> float:
        """Calculate form quality score."""
        angle_score = 1.0 - min(abs(angle - self.last_angle) / 180.0, 1.0) if self.last_angle else 1.0
        visibility_scores = [lm.visibility for lm in landmarks[:33] if hasattr(lm, 'visibility')]
        stability_score = np.mean(visibility_scores) if visibility_scores else 0.8
        form_score = (angle_score * 0.6) + (stability_score * 0.4)
        return max(0.0, min(1.0, form_score))


class ImprovedGenericCounter:
    """Enhanced rep counter with simplified stage transitions (up/down/neutral only)"""

    def __init__(
        self,
        indices: Tuple[int, int, int],
        up_threshold: float,
        down_threshold: float,
        ref_angle_patterns=None,
        accuracy_threshold: float = 0.4,
        angle_tolerance: float = 25.0,
        cooldown: float = 1.5,
        min_range: float = 30.0
    ):
        self.indices = indices
        self.up_threshold = max(up_threshold, down_threshold + 15)
        self.down_threshold = min(down_threshold, up_threshold - 15)
        self.ref_angle_patterns = ref_angle_patterns or []
        self.accuracy_threshold = accuracy_threshold
        self.angle_tolerance = angle_tolerance
        self.cooldown = cooldown
        self.min_range = min_range

        # State
        self.stage = "neutral"
        self.count = 0
        self.last_rep_time = 0
        self.last_angle = 0.0
        self.angle_history = deque(maxlen=5)
        self.initialized = False

        # Movement tracking
        self.peak_angle = 0.0
        self.valley_angle = 180.0
        self.movement_quality = 0.0

        # Trajectory
        self.trajectory = TrajectoryTracker()

        # Debug info
        self.debug_info = {
            "last_angle": 0,
            "last_stage": "neutral",
            "last_similarity": 0,
            "stage_changes": [],
            "peak_angle": 0,
            "valley_angle": 180,
            "movement_range": 0
        }

    def _calculate_angle(self, a, b, c) -> float:
        """Calculate joint angle safely"""
        try:
            if hasattr(a, 'x'):
                a_pos = np.array([a.x, a.y])
                b_pos = np.array([b.x, b.y])
                c_pos = np.array([c.x, c.y])
            else:
                a_pos = np.array(a)
                b_pos = np.array(b)
                c_pos = np.array(c)

            ba = a_pos - b_pos
            bc = c_pos - b_pos

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return float(np.degrees(angle))
        except Exception as e:
            print(f"Angle calculation error: {e}")
            return 0.0

    def _smooth_angle(self, angle: float) -> float:
        """Apply exponential smoothing to angle"""
        if not self.angle_history:
            self.angle_history.append(angle)
            return angle
        
        alpha = 0.3
        smoothed = alpha * angle + (1 - alpha) * self.angle_history[-1]
        self.angle_history.append(smoothed)
        return float(smoothed)

    def _calculate_similarity(self, angle: float) -> float:
        """Calculate similarity to reference angle patterns using Gaussian RBF"""
        if not self.ref_angle_patterns:
            return 1.0  # Perfect match if no reference

        sigma = 15.0  # Gaussian standard deviation
        similarities = []
        for ref_angle in self.ref_angle_patterns:
            similarity = np.exp(-((angle - ref_angle) ** 2) / (2 * sigma ** 2))
            similarities.append(similarity)

        return float(max(similarities)) if similarities else 0.0

    def _is_valid_form(self, angle: float) -> Tuple[bool, float]:
        """Check if form is valid based on angle and reference patterns"""
        similarity = self._calculate_similarity(angle)
        is_valid = similarity >= self.accuracy_threshold
        return is_valid, similarity

    def _update_movement_tracking(self, angle: float):
        """Update peak and valley angles for range of motion"""
        if angle < self.valley_angle:
            self.valley_angle = angle
        if angle > self.peak_angle:
            self.peak_angle = angle

    def _detect_rep_completion(self, angle: float, current_time: float, similarity: float) -> bool:
        """Detect rep completion via angle thresholds and form quality"""
        current_range = self.peak_angle - self.valley_angle
        
        if current_range < self.min_range:
            return False

        # State machine for 3-stage movement
        new_stage = None
        if angle >= self.up_threshold:
            new_stage = "up"
        elif angle <= self.down_threshold:
            new_stage = "down"
        else:
            new_stage = self.stage

        # Rep completed on cycle: down -> up -> down
        rep_completed = (
            self.stage == "up" and new_stage == "down" and
            self._validate_rep_completion(current_time, similarity)
        )

        self.stage = new_stage
        self.last_angle = angle

        return rep_completed

    def _validate_rep_completion(self, current_time: float, similarity: float) -> bool:
        """Validate rep completion with cooldown and quality checks"""
        if current_time - self.last_rep_time < self.cooldown:
            return False
        
        # Accept rep even if form isn't perfect, but track quality
        quality_check = similarity >= max(0.2, self.accuracy_threshold * 0.5)
        return quality_check

    def analyze_landmarks(self, landmarks, only_count_if_matching=False, timestamp: float = None) -> Tuple[int, float, float]:
        """Main analysis method - returns (rep_count, angle, similarity)"""
        try:
            current_time = timestamp if timestamp is not None else time.time()
            a_idx, b_idx, c_idx = self.indices
            
            if any(idx >= len(landmarks) for idx in [a_idx, b_idx, c_idx]):
                return self.count, 0.0, 0.0

            # Calculate and smooth angle
            raw_angle = self._calculate_angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
            angle = self._smooth_angle(raw_angle)

            # Calculate similarity
            similarity = self._calculate_similarity(angle)

            # Update trajectory
            self.trajectory.add_point(angle, current_time)
            self._update_movement_tracking(angle)

            # Detect rep completion
            if self._detect_rep_completion(angle, current_time, similarity):
                self.count += 1
                self.last_rep_time = current_time
                self.trajectory.mark_rep_completion()
                print(f"[Rep Completed] Total: {self.count}, Angle: {angle:.1f}°, Similarity: {similarity:.2f}")

            # Update debug info
            self.debug_info.update({
                "last_angle": angle,
                "last_stage": self.stage,
                "last_similarity": similarity,
                "peak_angle": self.peak_angle,
                "valley_angle": self.valley_angle,
                "movement_range": self.peak_angle - self.valley_angle
            })

            return self.count, angle, similarity

        except Exception as e:
            print(f"analyze_landmarks error: {e}")
            return self.count, 0.0, 0.0

    def get_trajectory_data(self) -> Dict[str, Any]:
        """Get trajectory data for visualization"""
        return self.trajectory.get_current_trajectory()

    def reset(self):
        """Reset counter for new exercise"""
        self.stage = "neutral"
        self.count = 0
        self.last_rep_time = 0
        self.last_angle = 0.0
        self.angle_history.clear()
        self.peak_angle = 0.0
        self.valley_angle = 180.0
        self.trajectory.reset()
        self.debug_info = {
            "last_angle": 0,
            "last_stage": "neutral",
            "last_similarity": 0,
            "stage_changes": [],
            "peak_angle": 0,
            "valley_angle": 180,
            "movement_range": 0
        }

    def get_debug_info(self) -> Dict[str, Any]:
        """Get comprehensive debug information"""
        trajectory_data = self.trajectory.get_current_trajectory()
        return {
            **self.debug_info,
            "trajectory_points": len(trajectory_data['angles']),
            "recent_range": self.trajectory.get_recent_range(),
            "thresholds": {
                "up": self.up_threshold,
                "down": self.down_threshold
            }
        }


class SixPointRepCounter:
    """Advanced rep counter with 6-stage progression tracking"""

    def __init__(
        self,
        indices: Tuple[int, int, int],
        up_threshold: float = 160.0,
        down_threshold: float = 50.0,
        angle_tolerance: float = 10.0,
        cooldown: float = 1.5,
        warmup_frames: int = 30,
        min_frames_per_stage: int = 3
    ):
        self.indices = indices
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.angle_tolerance = angle_tolerance
        self.cooldown = cooldown
        self.warmup_frames = warmup_frames
        self.min_frames_per_stage = min_frames_per_stage

        # 6-stage progression
        self.stages = ["min", "mid_up", "max", "mid_down", "min_return", "min"]
        self.stage_index = 0
        self.frames_in_stage = 0
        self.frame_count = 0

        # Adaptive thresholds
        self.min_angle = down_threshold
        self.max_angle = up_threshold
        self.mid_angle = (self.min_angle + self.max_angle) / 2

        # Tracking
        self.count = 0
        self.last_rep_time = 0
        self.last_angle = 0.0
        self.angle_history = deque(maxlen=5)
        self.initialized = False

        # Adaptive learning
        self.observed_min = float('inf')
        self.observed_max = float('-inf')

        # Trajectory
        self.trajectory = TrajectoryTracker()

        # Debug
        self.debug_info = {
            "current_stage": "min",
            "rep_count": 0,
            "last_angle": 0,
            "stage_progress": [],
            "movement_range": 0
        }

    def _calculate_angle(self, a, b, c) -> float:
        """Calculate joint angle"""
        try:
            if hasattr(a, 'x'):
                a_pos = np.array([a.x, a.y])
                b_pos = np.array([b.x, b.y])
                c_pos = np.array([c.x, c.y])
            else:
                a_pos = np.array(a)
                b_pos = np.array(b)
                c_pos = np.array(c)

            ba = a_pos - b_pos
            bc = c_pos - b_pos

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return float(np.degrees(angle))
        except:
            return 0.0

    def _smooth_angle(self, angle: float) -> float:
        """Exponential moving average smoothing"""
        if not self.angle_history:
            self.angle_history.append(angle)
            return angle
        alpha = 0.3
        smoothed = alpha * angle + (1 - alpha) * self.angle_history[-1]
        self.angle_history.append(smoothed)
        return float(smoothed)

    def _update_adaptive_angles(self, angle: float):
        """Learn adaptive thresholds from observed movement"""
        if angle < self.observed_min:
            self.observed_min = angle
        if angle > self.observed_max:
            self.observed_max = angle

        if (self.frame_count >= self.warmup_frames and not self.initialized and
            self.observed_max - self.observed_min > 20):
            self.min_angle = self.observed_min - 5
            self.max_angle = self.observed_max + 5
            self.mid_angle = (self.min_angle + self.max_angle) / 2
            self.debug_info["adaptive_angles_set"] = True
            print(f"[Adaptive] Min: {self.min_angle:.1f}°, Max: {self.max_angle:.1f}°")
        
        self.frame_count += 1

    def _check_stage_transition(self, angle: float) -> bool:
        """Check if should advance to next stage"""
        tolerance = self.angle_tolerance
        current_stage = self.stages[self.stage_index]
        next_stage_idx = (self.stage_index + 1) % len(self.stages)
        next_stage = self.stages[next_stage_idx]

        if current_stage == "min" and next_stage == "mid_up":
            return angle > self.min_angle + tolerance
        elif current_stage == "mid_up" and next_stage == "max":
            return angle >= self.max_angle - tolerance
        elif current_stage == "max" and next_stage == "mid_down":
            return angle < self.max_angle - tolerance
        elif current_stage == "mid_down" and next_stage == "min_return":
            return angle <= self.min_angle + tolerance
        elif current_stage == "min_return" and next_stage == "min":
            return angle <= self.min_angle + tolerance and self.stage_index > 0
        return False

    def _advance_stage(self, angle: float, current_time: float) -> bool:
        """Advance to next stage and return if rep completed"""
        rep_completed = False
        old_stage = self.stages[self.stage_index]
        self.stage_index = (self.stage_index + 1) % len(self.stages)
        new_stage = self.stages[self.stage_index]
        self.frames_in_stage = 0

        if old_stage == "min_return" and new_stage == "min":
            if current_time - self.last_rep_time >= self.cooldown:
                self.count += 1
                self.last_rep_time = current_time
                self.trajectory.mark_rep_completion()
                rep_completed = True
                print(f"[Rep Counted] Total: {self.count}, Angle: {angle:.1f}°")

        return rep_completed

    def analyze_landmarks(self, landmarks) -> Tuple[int, float, float]:
        """Analyze landmarks with 6-point trajectory"""
        try:
            current_time = time.time()
            a_idx, b_idx, c_idx = self.indices
            
            if any(idx >= len(landmarks) for idx in [a_idx, b_idx, c_idx]):
                return self.count, 0.0, 0.0

            raw_angle = self._calculate_angle(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
            angle = self._smooth_angle(raw_angle)
            self.last_angle = angle
            self.trajectory.add_point(angle, current_time)
            self._update_adaptive_angles(angle)

            if not self.initialized and self.frame_count >= self.warmup_frames:
                if angle <= self.min_angle + self.angle_tolerance:
                    self.stage_index = 0
                elif angle >= self.max_angle - self.angle_tolerance:
                    self.stage_index = 2
                else:
                    self.stage_index = 1
                self.initialized = True
                print(f"[Initialized] Stage: {self.stages[self.stage_index]} at {angle:.1f}°")
                return self.count, angle, 0.5

            if self.frame_count < self.warmup_frames:
                return self.count, angle, 0.0

            if self._check_stage_transition(angle):
                self.frames_in_stage += 1
                if self.frames_in_stage >= self.min_frames_per_stage:
                    self._advance_stage(angle, current_time)
                    self.frames_in_stage = 0
            else:
                self.frames_in_stage = 0

            current_stage = self.stages[self.stage_index]
            movement_range = self.max_angle - self.min_angle
            
            self.debug_info.update({
                "current_stage": current_stage,
                "rep_count": self.count,
                "last_angle": angle,
                "movement_range": movement_range,
            })

            return self.count, angle, 0.5

        except Exception as e:
            print(f"analyze_landmarks error: {e}")
            return self.count, 0.0, 0.0

    def get_trajectory_data(self) -> Dict[str, Any]:
        return self.trajectory.get_current_trajectory()

    def reset(self):
        """Reset counter"""
        self.stage_index = 0
        self.count = 0
        self.last_rep_time = 0
        self.last_angle = 0.0
        self.angle_history.clear()
        self.initialized = False
        self.trajectory.reset()

    def get_debug_info(self) -> Dict[str, Any]:
        trajectory_data = self.trajectory.get_current_trajectory()
        return {
            **self.debug_info,
            "stage_sequence": self.stages,
            "current_stage_index": self.stage_index,
            "trajectory_points": len(trajectory_data['angles']),
            "recent_range": self.trajectory.get_recent_range(),
            "thresholds": {
                "min": self.min_angle,
                "mid": self.mid_angle,
                "max": self.max_angle
            }
        }


class SmartRepCounter:
    """Router that delegates to appropriate counter based on exercise"""

    def __init__(self):
        self.exercise_counters: Dict[str, ImprovedGenericCounter] = {}
        self.total_counts: Dict[str, int] = {}

    def analyze_frame(self, exercise: str, landmarks, is_matching_reference=True, timestamp: float = None) -> Tuple[int, float, float]:
        """Analyze a single frame"""
        exercise = exercise.lower()
        counter = self.exercise_counters.get(exercise)

        # Create default counter if missing
        if not counter:
            default_indices = (
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
            )
            counter = ImprovedGenericCounter(
                default_indices,
                up_threshold=160.0,
                down_threshold=50.0,
                accuracy_threshold=0.3,
                angle_tolerance=20.0,
                min_range=25.0
            )
            self.exercise_counters[exercise] = counter

        count, angle, similarity = counter.analyze_landmarks(landmarks, only_count_if_matching=False, timestamp=timestamp)
        self.total_counts[exercise] = count

        return count, angle, similarity

    def register_exercise_from_rule(self, exercise: str, rule: Dict[str, Any]):
        """Register exercise with server-provided thresholds"""
        exercise = exercise.lower()
        try:
            indices = tuple(rule.get("indices", []))
            if len(indices) != 3:
                raise ValueError(f"Invalid indices: {indices}")

            up = float(rule.get("up_threshold", 160.0))
            down = float(rule.get("down_threshold", 50.0))
            ref_angles = rule.get("angles", [])

            if up <= down:
                up, down = down + 20, up

            counter = ImprovedGenericCounter(
                indices,
                up_threshold=up,
                down_threshold=down,
                ref_angle_patterns=ref_angles,
                accuracy_threshold=0.3,
                angle_tolerance=20.0,
                min_range=max(20.0, (up - down) * 0.3)
            )
            self.exercise_counters[exercise] = counter
            self.total_counts[exercise] = 0

        except Exception as e:
            print(f"Failed to register {exercise}: {e}")
            self.exercise_counters[exercise] = ImprovedGenericCounter(
                (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                 mp_pose.PoseLandmark.LEFT_ELBOW.value,
                 mp_pose.PoseLandmark.LEFT_WRIST.value),
                160.0, 50.0
            )

    def get_total_count(self, exercise: str) -> int:
        return self.total_counts.get(exercise.lower(), 0)

    def get_trajectory_data(self, exercise: str) -> Dict[str, Any]:
        """Return trajectory data"""
        exercise = exercise.lower()
        counter = self.exercise_counters.get(exercise)
        if counter:
            return counter.get_trajectory_data()
        return {"angles": [], "timestamps": [], "velocities": [], "rep_boundaries": []}

    def reset_exercise(self, exercise: str):
        exercise = exercise.lower()
        if exercise in self.exercise_counters:
            self.exercise_counters[exercise].reset()
            self.total_counts[exercise] = 0

    def reset_all(self):
        for counter in self.exercise_counters.values():
            counter.reset()
        self.total_counts.clear()

    def get_debug_info(self, exercise: str) -> Dict[str, Any]:
        exercise = exercise.lower()
        counter = self.exercise_counters.get(exercise)
        if counter:
            return counter.get_debug_info()
        return {"error": f"No counter found for {exercise}"}

    def print_debug_summary(self, exercise: str):
        """Print detailed debug summary"""
        debug_info = self.get_debug_info(exercise)
        print(f"\n=== DEBUG INFO for {exercise.upper()} ===")
        print(f"Rep Count: {debug_info.get('rep_count', 0)}")
        print(f"Last Angle: {debug_info.get('last_angle', 0):.1f} deg")
        print(f"Movement Range: {debug_info.get('movement_range', 0):.1f} deg")
        print(f"Recent Range: {debug_info.get('recent_range', 0):.1f} deg")
        print(f"Trajectory Points: {debug_info.get('trajectory_points', 0)}")



