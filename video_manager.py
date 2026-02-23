"""
Video management and reference video analysis.
"""

import os
import shutil
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional, BinaryIO

from models import VideoAnalysisResult
from rep_counters import ExerciseCounter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# Will be set by Main.py
LABELS = []


class VideoManager:
    """Manages video upload, storage, and reference extraction"""
    
    def __init__(self, upload_dir: str = "uploaded_videos"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        self.videos: Dict[str, Dict[str, Any]] = {}
        self.analysis_cache: Dict[str, VideoAnalysisResult] = {}
        self.reference_videos: Dict[str, VideoAnalysisResult] = {}
        self.supported_exercises: set = set()
        
    def save_video(self, exercise_type: str, video_file: BinaryIO, original_filename: str) -> Dict[str, Any]:
        """Save uploaded video and extract metadata"""
        try:
            exercise_type = exercise_type.lower()
            timestamp = int(time.time())
            filename = f"{exercise_type}_{timestamp}_{original_filename}"
            filepath = os.path.join(self.upload_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(video_file.read())
            
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            video_info = {
                "id": f"{exercise_type}_{timestamp}",
                "filename": filename,
                "filepath": filepath,
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "upload_time": timestamp,
                "file_size": os.path.getsize(filepath)
            }
            
            self.videos[video_info["id"]] = video_info
            self.supported_exercises.add(exercise_type)
            return video_info
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def analyze_reference_video(self, video_id: str, target_exercise: str) -> VideoAnalysisResult:
        """Analyze reference video and extract exercise patterns"""
        target_exercise = target_exercise.lower()
        if video_id in self.analysis_cache:
            return self.analysis_cache[video_id]
        
        video_info = self.videos.get(video_id)
        if not video_info:
            raise ValueError("Video not found")
        
        cap = cv2.VideoCapture(video_info["filepath"])
        
        poses = []
        form_scores = []
        frames_analyzed = 0
        
        # Candidate joints for angle calculation (include elbows, knees, shoulders, hips)
        candidate_joints = {
            "left_elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                           mp_pose.PoseLandmark.LEFT_ELBOW.value,
                           mp_pose.PoseLandmark.LEFT_WRIST.value),
            "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.RIGHT_WRIST.value),
            "left_knee": (mp_pose.PoseLandmark.LEFT_HIP.value,
                          mp_pose.PoseLandmark.LEFT_KNEE.value,
                          mp_pose.PoseLandmark.LEFT_ANKLE.value),
            "right_knee": (mp_pose.PoseLandmark.RIGHT_HIP.value,
                           mp_pose.PoseLandmark.RIGHT_KNEE.value,
                           mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            "left_shoulder": (mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value),
            "right_shoulder": (mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                               mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                               mp_pose.PoseLandmark.RIGHT_HIP.value),
            "left_hip": (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_HIP.value,
                         mp_pose.PoseLandmark.LEFT_KNEE.value),
            "right_hip": (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                          mp_pose.PoseLandmark.RIGHT_HIP.value,
                          mp_pose.PoseLandmark.RIGHT_KNEE.value),
        }
        angle_traces = {k: [] for k in candidate_joints.keys()}
        
        frame_interval = max(1, int(video_info["fps"] / 10))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    keypoints = [lm.x for lm in landmarks[:33]] + [lm.y for lm in landmarks[:33]]
                    poses.append(keypoints)
                    
                    # Compute candidate joint angles
                    for k, idxs in candidate_joints.items():
                        try:
                            a = landmarks[idxs[0]]
                            b = landmarks[idxs[1]]
                            c = landmarks[idxs[2]]
                            angle = ExerciseCounter.calculate_angle(a, b, c)
                            angle_traces[k].append(angle)
                        except Exception:
                            angle_traces[k].append(0.0)
                    
                    frames_analyzed += 1
            frame_idx += 1
        
        cap.release()
        
        # Compute per-joint cleaned traces, deltas, thresholds and similarity
        joint_thresholds = {}
        cleaned_traces = {}
        deltas = {}
        joint_similarity = {}

        NOISE_THRESHOLD = 10.0  # degrees: ignore changes smaller than this

        for joint, trace in angle_traces.items():
            arr = np.array(trace) if trace else np.array([])
            if arr.size == 0:
                joint_thresholds[joint] = {"min": 0.0, "mid": 0.0, "max": 0.0}
                cleaned_traces[joint] = []
                deltas[joint] = []
                joint_similarity[joint] = 0.0
                continue

            # Raw absolute deltas
            raw_d = np.abs(np.diff(arr, prepend=arr[0]))

            # Clean small fluctuations: if delta < NOISE_THRESHOLD, keep previous value
            cleaned = arr.copy()
            for i in range(1, len(cleaned)):
                if raw_d[i] < NOISE_THRESHOLD:
                    cleaned[i] = cleaned[i-1]

            cleaned_d = np.abs(np.diff(cleaned, prepend=cleaned[0]))

            # Thresholds via percentiles on cleaned trace
            min_th = float(np.percentile(cleaned, 20))
            mid_th = float(np.percentile(cleaned, 50))
            max_th = float(np.percentile(cleaned, 80))

            joint_thresholds[joint] = {"min": min_th, "mid": mid_th, "max": max_th}
            cleaned_traces[joint] = cleaned.tolist()
            deltas[joint] = cleaned_d.tolist()

            # Similarity/movement score: fraction of frames with cleaned movement
            movement_fraction = float((cleaned_d > 0).sum()) / max(1, len(cleaned_d))
            std_score = min(1.0, float(np.std(cleaned)) / 30.0)
            joint_similarity[joint] = float((movement_fraction * 0.6) + (std_score * 0.4))

        # Select best joint by movement score (stddev * similarity)
        scores = {j: (np.std(np.array(angle_traces[j])) if angle_traces[j] else 0.0) * joint_similarity.get(j, 0.0) for j in angle_traces.keys()}
        best_joint = max(scores.keys(), key=lambda k: scores[k]) if scores else None

        # Partition joints into moving vs static based on similarity score
        moving_joints = []
        static_joints = []
        for joint in angle_traces.keys():
            sim = joint_similarity.get(joint, 0.0)
            std = float(np.std(np.array(angle_traces[joint])) if angle_traces[joint] else 0.0)
            if sim >= 0.15 and std > 5.0:
                moving_joints.append(joint)
            else:
                static_joints.append(joint)

        # Compile a single list of strict landmark indices from all static joints
        strict_joint_indices = []
        for joint_name in static_joints:
            if joint_name in candidate_joints:
                strict_joint_indices.extend(candidate_joints[joint_name])
        strict_joint_indices = sorted(list(set(strict_joint_indices)))

        # Build rep_rules list for joints that show movement
        rep_rules_list = []
        for joint in moving_joints:
            rep_rules_list.append({
                "joint_name": joint,
                "indices": candidate_joints[joint],
                "up_threshold": joint_thresholds[joint]["max"],
                "down_threshold": joint_thresholds[joint]["min"],
                "similarity": joint_similarity[joint],
                "strict_indices": strict_joint_indices,
                "strict_joint_names": static_joints
            })

        # If no joints passed threshold, include best joint as fallback
        if not rep_rules_list and best_joint:
            # If best_joint is the only moving one, all others are static
            static_joints = [j for j in angle_traces.keys() if j != best_joint]
            strict_joint_indices = []
            for j_name in static_joints:
                if j_name in candidate_joints:
                    strict_joint_indices.extend(candidate_joints[j_name])
            strict_joint_indices = sorted(list(set(strict_joint_indices)))
            
            rep_rules_list.append({
                "joint_name": best_joint,
                "indices": candidate_joints[best_joint],
                "up_threshold": joint_thresholds[best_joint]["max"],
                "down_threshold": joint_thresholds[best_joint]["min"],
                "similarity": joint_similarity.get(best_joint, 0.0),
                "strict_indices": strict_joint_indices,
                "strict_joint_names": static_joints
            })

        # Estimate reps per joint and sum them so reference shows all joints' reps
        total_est_count = 0
        per_joint_estimates = {}
        for rule in rep_rules_list:
            jname = rule.get("joint_name")
            down_threshold = rule.get("down_threshold", 0.0)
            up_threshold = rule.get("up_threshold", 0.0)
            stage = "neutral"
            est = 0
            cleaned_best = cleaned_traces.get(jname, [])
            for ang in cleaned_best:
                try:
                    if ang < down_threshold and stage != "down":
                        stage = "down"
                    elif ang > up_threshold and stage == "down":
                        stage = "up"
                        est += 1
                except Exception:
                    continue
            per_joint_estimates[jname] = int(est)
            total_est_count += int(est)

        result = VideoAnalysisResult(
            total_reps=total_est_count,
            average_form_score=0.0,
            exercise_detected=target_exercise,
            key_poses=poses,
            duration=video_info["duration"],
            analysis_frames=frames_analyzed,
            rep_rule=rep_rules_list[0] if rep_rules_list else None,
            angle_traces_by_joint=angle_traces,
            angle_deltas_by_joint=deltas,
            cleaned_traces_by_joint=cleaned_traces,
            joint_thresholds=joint_thresholds,
            joint_similarity=joint_similarity,
            rep_rules=rep_rules_list,
            per_joint_estimates=per_joint_estimates
        )
        
        # Cache and register reference
        self.analysis_cache[video_id] = result
        self.reference_videos[target_exercise] = result
        self.supported_exercises.add(target_exercise)
        
        # Register exercise type
        if target_exercise not in LABELS:
            LABELS.append(target_exercise)
            print(f"[INFO] Added {target_exercise} to exercise labels")
        
        return result
    
    def get_reference_video(self, exercise_type: str) -> Optional[VideoAnalysisResult]:
        """Get reference video analysis for an exercise"""
        return self.reference_videos.get(exercise_type.lower())
