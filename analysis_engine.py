"""
Real-time frame analysis engine for exercise form detection and rep counting.
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, List, Any

from models import ExerciseAnalysis
from rep_counters import SmartRepCounter, ImprovedGenericCounter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


class RealTimeAnalysisEngine:
    """Core processing logic for real-time exercise analysis"""
    
    def __init__(self, model, labels, video_manager, session_id: str, target_exercise: str):
        self.model = model
        self.labels = labels or []
        self.video_manager = video_manager
        self.session_id = session_id
        self.target_exercise = target_exercise.lower()
        self.pose_buffer = []
        self.rep_counter = SmartRepCounter()
        # Process every incoming frame (no frame skipping)
        self.last_analysis_time = 0
        self.analysis_interval = 0.05
        self.last_active = time.time()
        self.session_timeout = 3600
        self.min_buffer_size = 3
        self.max_buffer_size = 10
        self.last_exercise_prediction = None
        self.prediction_cache_duration = 0.5
        self.last_prediction_time = 0
        
        # Debug tracking
        self.debug_mode = True
        self.frame_count = 0
        
        # Per-joint counters discovered from reference
        self.per_joint_counters = {}
        self.rep_rules = []
        self._register_target_exercise()

    def _register_target_exercise(self):
        """Register the target exercise with proper error handling"""
        if self.video_manager:
            reference = self.video_manager.get_reference_video(self.target_exercise)
            # Support multiple rep rules (per-joint) produced by reference analysis
            if reference and getattr(reference, 'rep_rules', None):
                temp_counters = {}
                for rule in reference.rep_rules:
                    try:
                        indices = tuple(rule.get('indices', []))
                        if len(indices) != 3:
                            continue
                        up = float(rule.get('up_threshold', 160.0))
                        down = float(rule.get('down_threshold', 50.0))
                        ref_angles = rule.get('angles', [])

                        # Create a dedicated counter per joint and keep state across frames
                        counter = ImprovedGenericCounter(
                            indices=indices,
                            up_threshold=max(up, down + 15),
                            down_threshold=min(down, up - 15),
                            ref_angle_patterns=ref_angles,
                            accuracy_threshold=0.3,
                            angle_tolerance=20.0,
                            min_range=max(20.0, (up - down) * 0.3),
                            strict_joint_indices=rule.get("strict_indices", [])
                        )
                        name = rule.get('joint_name') or rule.get('name') or f"{indices}"
                        temp_counters[name] = counter
                        self.rep_rules.append(rule)
                    except Exception as e:
                        print(f"[WARN] Failed to create per-joint counter: {e}")

                # Filter counters to only include pairs (left/right)
                all_joint_names = list(temp_counters.keys())
                joints_to_keep = set()
                base_joints = set()

                for name in all_joint_names:
                    if name.startswith("left_"):
                        base_joints.add(name[5:])
                    elif name.startswith("right_"):
                        base_joints.add(name[6:])
                    else:
                        joints_to_keep.add(name) # Keep non-paired joints

                for base in base_joints:
                    left_name = f"left_{base}"
                    right_name = f"right_{base}"
                    if left_name in temp_counters and right_name in temp_counters:
                        joints_to_keep.add(left_name)
                        joints_to_keep.add(right_name)
                
                self.per_joint_counters = {name: temp_counters[name] for name in joints_to_keep}

                if self.per_joint_counters:
                    print(f"[INFO] Registered {len(self.per_joint_counters)} filtered per-joint counters for {self.target_exercise}")
                    return

            # Fallback: single rule registration if available
            if reference and reference.rep_rule:
                self.rep_counter.register_exercise_from_rule(self.target_exercise, reference.rep_rule)
                print(f"[INFO] Registered {self.target_exercise} with reference data")
            else:
                print(f"[INFO] No reference found for {self.target_exercise}, using fallback")

    def _decode_and_resize_frame(self, image_data: bytes):
        """Decode base64 image data and resize"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                if frame.shape[1] > 640:
                    scale = 640 / frame.shape[1]
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            return frame
        except Exception as e:
            print(f"[ERROR] Error decoding frame: {e}")
            return None

    def _create_no_pose_analysis(self) -> ExerciseAnalysis:
        """Create analysis when no pose detected"""
        return ExerciseAnalysis(
            exercise_name=self.target_exercise,
            confidence=0.0,
            rep_count=self.rep_counter.get_total_count(self.target_exercise),
            current_angle=0.0,
            form_feedback="No pose detected - ensure full body is visible",
            similarity_score=0.0,
            stage="neutral",
            timestamp=time.time(),
            is_matching_reference=False,
            pose_landmarks=None
        )

    def _classify_exercise_optimized(self, current_time: float) -> Tuple[str, float]:
        """Return target exercise with high confidence"""
        return self.target_exercise, 0.95

    def _calculate_similarity_score(self, landmarks) -> float:
        """Calculate similarity with reference poses"""
        if not self.video_manager:
            return 0.8
            
        reference = self.video_manager.get_reference_video(self.target_exercise)
        if not reference or not reference.key_poses or not self.pose_buffer:
            return 0.8
        
        try:
            current_pose = self.pose_buffer[-1]
            similarities = []
            for ref_pose in reference.key_poses[:3]:
                if len(ref_pose) == len(current_pose):
                    diff = np.mean(np.abs(np.array(ref_pose) - np.array(current_pose)))
                    similarity = max(0, 1 - (diff * 2))
                    similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.8
            
        except Exception as e:
            print(f"[ERROR] Similarity calculation error: {e}")
            return 0.7

    def _generate_form_feedback(self, form_score: float, is_matching: bool, stage: str, angle: float) -> str:
        """Generate form feedback"""
        if not is_matching:
            return f"Exercise mismatch - expecting {self.target_exercise.replace('_', ' ')}"
        
        feedback_parts = []
        
        if form_score > 0.85:
            feedback_parts.append("Excellent form!")
        elif form_score > 0.7:
            feedback_parts.append("Good form")
        elif form_score > 0.5:
            feedback_parts.append("Moderate form - focus on technique")
        else:
            feedback_parts.append("Poor form - check your posture")
        
        if stage == "down":
            feedback_parts.append("Hold the position")
        elif stage == "up":
            feedback_parts.append("Full extension")
        else:
            feedback_parts.append("Ready position")
            
        counter = self.rep_counter.exercise_counters.get(self.target_exercise)
        if counter:
            if angle < counter.down_threshold - 10:
                feedback_parts.append("Too low")
            elif angle > counter.up_threshold + 10:
                feedback_parts.append("Too high")
        
        return " • ".join(feedback_parts)

    def _extract_key_landmarks(self, pose_landmarks) -> List[Dict]:
        """Extract key landmarks for visualization"""
        if not pose_landmarks:
            return []
        
        key_indices = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        landmarks = []
        for idx in key_indices:
            lm = pose_landmarks.landmark[idx.value]
            landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z) if hasattr(lm, 'z') else 0.0,
                "visibility": float(lm.visibility) if hasattr(lm, 'visibility') else 1.0,
                "name": idx.name
            })
        
        return landmarks

    def analyze_frame(self, image_data: bytes) -> Optional[ExerciseAnalysis]:
        """Main frame analysis method"""
        current_time = time.time()
        self.last_active = current_time
        self.frame_count += 1

        # No frame skipping: process each incoming frame (rate limited by analysis_interval)

        # Rate limiting
        if current_time - self.last_analysis_time < self.analysis_interval:
            return None

        # Decode frame
        frame = self._decode_and_resize_frame(image_data)
        if frame is None:
            return None

        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return self._create_no_pose_analysis()

        landmarks = results.pose_landmarks.landmark
        
        # Build pose vector
        keypoints = []
        for lm in landmarks[:33]:
            keypoints.extend([lm.x, lm.y])

        # Update buffer
        self.pose_buffer.append(keypoints)
        if len(self.pose_buffer) > self.max_buffer_size:
            self.pose_buffer.pop(0)

        # Exercise classification
        if len(self.pose_buffer) >= self.min_buffer_size:
            detected_exercise, confidence = self._classify_exercise_optimized(current_time)
        else:
            detected_exercise = self.target_exercise
            confidence = 0.5

        is_matching_reference = (detected_exercise.lower() == self.target_exercise.lower())
        stuck_joints = None

        # Rep counting - support per-joint counters if available
        try:
            per_joint_results = {}
            primary_angle = 0.0
            primary_similarity = 0.0

            if self.per_joint_counters:
                angles = []
                similarities = []
                rep_counts = []
                for name, counter in self.per_joint_counters.items():
                    count, angle, similarity = counter.analyze_landmarks(landmarks)
                    per_joint_results[name] = {
                        "reps": int(count),
                        "angle": float(angle),
                        "similarity": float(similarity),
                        "stage": counter.stage,
                        "up_threshold": getattr(counter, 'up_threshold', None),
                        "down_threshold": getattr(counter, 'down_threshold', None),
                    }
                    rep_counts.append(int(count))
                    angles.append(float(angle))
                    similarities.append(float(similarity))

                rep_count = min(rep_counts) if rep_counts else 0
                max_reps = max(rep_counts) if rep_counts else 0

                if max_reps > rep_count:
                    stuck_joints = [name for name, result in per_joint_results.items() if result['reps'] == rep_count]

                # Use averages across all detected joints rather than picking a single "best" joint
                current_angle = float(np.mean(angles)) if angles else 0.0
                similarity_form_score = float(np.mean(similarities)) if similarities else 0.0

                if self.debug_mode and self.frame_count % 30 == 0:
                    print(f"[DEBUG] Frame {self.frame_count}: Per-joint counts: { {k:v['reps'] for k,v in per_joint_results.items()} }, avg_angle={current_angle:.1f}")

            else:
                rep_count, current_angle, similarity_form_score = self.rep_counter.analyze_frame(
                    self.target_exercise,
                    landmarks,
                    is_matching_reference=True
                )
                per_joint_results = None

        except Exception as e:
            print(f"[ERROR] Rep counting error: {e}")
            rep_count = self.rep_counter.get_total_count(self.target_exercise)
            current_angle = 0.0
            similarity_form_score = 0.0
            per_joint_results = None

        similarity_score = self._calculate_similarity_score(landmarks)
        pose_landmarks_data = self._extract_key_landmarks(results.pose_landmarks)
        
        counter = self.rep_counter.exercise_counters.get(self.target_exercise)
        stage = counter.stage if counter else "neutral"

        form_feedback = self._generate_form_feedback(
            similarity_form_score, is_matching_reference, stage, current_angle
        )

        self.last_analysis_time = current_time

        return ExerciseAnalysis(
            exercise_name=detected_exercise,
            confidence=confidence,
            rep_count=rep_count,
            current_angle=current_angle,
            form_feedback=form_feedback,
            similarity_score=similarity_score,
            stage=stage,
            timestamp=current_time,
            is_matching_reference=is_matching_reference,
            pose_landmarks=pose_landmarks_data,
            per_joint=per_joint_results,
            stuck_joints=stuck_joints
        )

    def reset_session(self):
        """Reset analysis session"""
        self.pose_buffer = []
        self.rep_counter.reset_exercise(self.target_exercise)
        self.last_exercise_prediction = None
        self.frame_count = 0
        print(f"[INFO] Session {self.session_id} reset for {self.target_exercise}")
        self._register_target_exercise()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics for debugging"""
        counter = self.rep_counter.exercise_counters.get(self.target_exercise)
        return {
            "session_id": self.session_id,
            "target_exercise": self.target_exercise,
            "frames_processed": self.frame_count,
            "current_reps": self.rep_counter.get_total_count(self.target_exercise),
            "has_counter": counter is not None,
            "debug_info": self.rep_counter.get_debug_info(self.target_exercise) if counter else None,
            "pose_buffer_size": len(self.pose_buffer),
            "last_active": self.last_active
        }
