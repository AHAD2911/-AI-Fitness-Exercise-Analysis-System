"""
Trajectory tracking for exercise analysis and visualization.
"""

import time
from typing import Dict, List
from collections import deque


class TrajectoryTracker:
    """Track angle trajectories for analysis and visualization"""
    
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.angles = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
        self.velocities = deque(maxlen=max_length)
        self.rep_boundaries = []  # Store indices where reps completed
        
    def add_point(self, angle: float, timestamp: float):
        """Add new angle point to trajectory"""
        self.angles.append(angle)
        self.timestamps.append(timestamp)
        
        # Calculate velocity if we have enough points
        if len(self.angles) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            if dt > 0:
                velocity = (self.angles[-1] - self.angles[-2]) / dt
            else:
                velocity = 0.0
        else:
            velocity = 0.0
        self.velocities.append(velocity)
    
    def mark_rep_completion(self):
        """Mark current position as rep completion"""
        if len(self.angles) > 0:
            self.rep_boundaries.append(len(self.angles) - 1)
    
    def get_current_trajectory(self) -> Dict[str, List]:
        """Get current trajectory data"""
        return {
            'angles': list(self.angles),
            'timestamps': list(self.timestamps),
            'velocities': list(self.velocities),
            'rep_boundaries': self.rep_boundaries.copy()
        }
    
    def get_recent_range(self, window: int = 20) -> float:
        """Get range of motion in recent window"""
        if len(self.angles) < 2:
            return 0.0
        recent_angles = list(self.angles)[-window:]
        return max(recent_angles) - min(recent_angles)
    
    def reset(self):
        """Reset all trajectory data"""
        self.angles.clear()
        self.timestamps.clear()
        self.velocities.clear()
        self.rep_boundaries.clear()
