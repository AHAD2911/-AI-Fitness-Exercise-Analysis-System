"""
API route endpoints for FastAPI application.
"""

import json
import base64
from fastapi import Query, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import cv2
import numpy as np
import mediapipe as mp

from models import ExerciseAnalysis


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)


def register_routes(app, connection_manager, video_manager, smart_rep_counter):
    """Register all API routes with FastAPI app"""
    
    @app.websocket("/ws/live-analysis/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str, exercise_type: str = Query(...)):
        """WebSocket endpoint for live real-time analysis"""
        # Check if exercise is supported
        if exercise_type.lower() not in video_manager.supported_exercises:
            await websocket.close(
                code=1008, 
                reason=f"Exercise '{exercise_type}' not registered. Upload reference video first."
            )
            return

        # Create session
        try:
            session_id = await connection_manager.connect(
                websocket, exercise_type, None, [], video_manager
            )
        except Exception as e:
            await websocket.close(code=1011, reason=f"Connection error: {str(e)}")
            return

        try:
            await connection_manager.send_personal_message({
                "type": "connection_established",
                "session_id": session_id,
                "exercise_type": exercise_type,
                "message": f"Ready for {exercise_type.replace('_', ' ')} analysis"
            }, session_id)

            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "frame":
                    try:
                        image_data = base64.b64decode(message["frame"])
                        analysis_engine = connection_manager.get_analysis_engine(session_id)

                        if analysis_engine:
                            result = analysis_engine.analyze_frame(image_data)
                            if result:
                                response = {
                                    "type": "analysis",
                                    "data": {
                                        "exercise": result.exercise_name,
                                        "confidence": round(result.confidence, 2),
                                        "reps": result.rep_count,
                                        "angle": round(result.current_angle, 1),
                                        "matching": result.is_matching_reference,
                                        "stage": result.stage,
                                        "form_feedback": result.form_feedback,
                                        "similarity_score": round(result.similarity_score, 2),
                                        "timestamp": result.timestamp,
                                        "landmarks": result.pose_landmarks if result.pose_landmarks else [],
                                        "per_joint": result.per_joint if getattr(result, 'per_joint', None) else {}
                                    }
                                }
                                await connection_manager.send_json(response, session_id)

                    except Exception as e:
                        print(f"[ERROR] Frame processing error: {e}")
                        continue

                elif message.get("type") == "reset":
                    analysis_engine = connection_manager.get_analysis_engine(session_id)
                    if analysis_engine:
                        analysis_engine.reset_session()

                    await connection_manager.send_personal_message({
                        "type": "reset_complete",
                        "message": "Counter reset successfully"
                    }, session_id)

        except WebSocketDisconnect:
            print(f"[INFO] WebSocket disconnected for session {session_id}")
            connection_manager.disconnect(session_id)
        except Exception as e:
            print(f"[ERROR] WebSocket error: {e}")
            connection_manager.disconnect(session_id)


    @app.post("/upload-reference-video")
    async def upload_reference_video(
        file: UploadFile = File(...),
        exercise_type: str = Query(..., description="Exercise type for the reference video")
    ):
        """Upload and analyze reference video for an exercise type"""
        exercise_type = exercise_type.lower()
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        try:
            video_info = video_manager.save_video(exercise_type, file)
            
            analysis_result = video_manager.analyze_reference_video(
                video_info["id"], exercise_type
            )
            
            return {
                "status": "success",
                "video_info": video_info,
                "exercise_type": exercise_type,
                "message": f"Reference video analyzed and registered for {exercise_type.replace('_', ' ')}",
                "analysis_summary": {
                    "estimated_reps": analysis_result.total_reps,
                    "rep_rule": analysis_result.rep_rule,
                    "rep_rules": analysis_result.rep_rules,
                    "per_joint_estimates": analysis_result.per_joint_estimates,
                    "joint_thresholds": analysis_result.joint_thresholds,
                    "angle_traces_by_joint": analysis_result.angle_traces_by_joint,
                    "cleaned_traces_by_joint": analysis_result.cleaned_traces_by_joint
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/analyze-frame")
    async def analyze_frame(
        file: UploadFile = File(...),
        exercise_type: str = Query(..., description="Exercise type to analyze")
    ):
        """Analyze a single image frame"""
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            contents = await file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if not results or not results.pose_landmarks:
                return JSONResponse({
                    "status": "no_pose",
                    "rep_count": smart_rep_counter.get_total_count(exercise_type),
                    "current_angle": 0.0,
                    "similarity_score": 0.0,
                    "form_feedback": "No pose detected"
                })

            landmarks = results.pose_landmarks.landmark

            try:
                count, angle, similarity = smart_rep_counter.analyze_frame(
                    exercise_type, landmarks, is_matching_reference=True
                )
            except Exception as e:
                print(f"analyze_frame counter error: {e}")
                count = smart_rep_counter.get_total_count(exercise_type)
                angle = 0.0
                similarity = 0.0

            feedback = "Good" if similarity >= 0.5 else "Needs Improvement"

            return {
                "status": "ok",
                "rep_count": count,
                "current_angle": float(round(angle, 2)),
                "similarity_score": float(round(similarity, 3)),
                "form_feedback": feedback
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/")
    async def root():
        """Health check and system status"""
        return {
            "message": "AI Exercise Analysis System",
            "status": "running",
            "supported_exercises": list(video_manager.supported_exercises),
        }
