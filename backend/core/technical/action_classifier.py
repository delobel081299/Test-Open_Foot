"""
Action classifier for technical analysis
"""

from typing import List, Dict, Tuple
import numpy as np

from backend.core.tracking.byte_tracker import Track
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class ActionClassifier:
    """Classify football actions from player movements"""
    
    def __init__(self):
        self.action_types = [
            "running", "walking", "jumping", "kicking",
            "passing", "shooting", "heading", "dribbling",
            "tackling", "goalkeeping", "celebrating", "standing"
        ]
    
    def classify_actions(
        self,
        tracks: List[Track],
        frames: List[np.ndarray]
    ) -> List[Dict]:
        """Classify actions for all tracked players"""
        
        actions = []
        
        for track in tracks:
            track_actions = self._classify_track_actions(track, frames)
            actions.extend(track_actions)
        
        logger.info(f"Classified {len(actions)} actions from {len(tracks)} tracks")
        return actions
    
    def _classify_track_actions(
        self,
        track: Track,
        frames: List[np.ndarray]
    ) -> List[Dict]:
        """Classify actions for a single track"""
        
        # Placeholder implementation
        # In a real implementation, this would use a trained model
        
        actions = []
        
        # Simple heuristic-based classification
        if len(track.positions) > 1:
            # Calculate movement speed
            speeds = []
            for i in range(1, len(track.positions)):
                dx = track.positions[i][0] - track.positions[i-1][0]
                dy = track.positions[i][1] - track.positions[i-1][1]
                speed = np.sqrt(dx**2 + dy**2)
                speeds.append(speed)
            
            avg_speed = np.mean(speeds) if speeds else 0
            
            # Classify based on speed
            if avg_speed < 2:
                action_type = "standing"
            elif avg_speed < 5:
                action_type = "walking"
            else:
                action_type = "running"
            
            actions.append({
                "track_id": track.track_id,
                "action_type": action_type,
                "confidence": 0.8,
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "average_speed": avg_speed
            })
        
        return actions