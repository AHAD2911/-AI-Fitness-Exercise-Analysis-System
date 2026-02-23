"""
WebSocket connection management for real-time analysis sessions.
"""

import time
from typing import Dict, Optional
from fastapi import WebSocket

from analysis_engine import RealTimeAnalysisEngine


class ConnectionManager:
    """Manages WebSocket connections and analysis engine instances"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.analysis_engines: Dict[str, 'RealTimeAnalysisEngine'] = {}
    
    async def connect(self, websocket: WebSocket, exercise_type: str, model=None, labels=None, video_manager=None):
        """Accept new WebSocket connection and create analysis engine"""
        await websocket.accept()
        session_id = f"session_{int(time.time())}_{id(websocket)}"
        self.active_connections[session_id] = websocket
        
        # Create dedicated analysis engine for this session
        self.analysis_engines[session_id] = RealTimeAnalysisEngine(
            model, labels, video_manager, session_id, exercise_type
        )
        
        print(f"[INFO] WebSocket connected: {session_id} for {exercise_type}")
        return session_id
    
    def disconnect(self, session_id: str):
        """Disconnect WebSocket and cleanup"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.analysis_engines:
            del self.analysis_engines[session_id]
        print(f"[INFO] WebSocket disconnected: {session_id}")
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            try:
                import json
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    def get_analysis_engine(self, session_id: str) -> Optional['RealTimeAnalysisEngine']:
        """Get analysis engine for session"""
        return self.analysis_engines.get(session_id)
    
    async def send_json(self, data: dict, session_id: str):
        """Send JSON data over WebSocket"""
        await self.send_personal_message(data, session_id)
