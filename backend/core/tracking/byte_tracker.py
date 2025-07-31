import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import lap

from backend.core.detection.yolo_detector import Detection
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    is_active: bool = True
    frames_since_update: int = 0
    start_frame: int = 0
    end_frame: int = 0
    
    def update(self, detection: Detection, frame_num: int):
        """Update track with new detection"""
        self.detections.append(detection)
        self.frames.append(frame_num)
        
        # Calculate center position
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.positions.append((center_x, center_y))
        
        self.frames_since_update = 0
        self.end_frame = frame_num
        
        if self.start_frame == 0:
            self.start_frame = frame_num
    
    def predict_position(self) -> Tuple[float, float]:
        """Predict next position based on motion history"""
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else (0, 0)
        
        # Simple linear prediction
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        
        predicted_x = self.positions[-1][0] + dx
        predicted_y = self.positions[-1][1] + dy
        
        return (predicted_x, predicted_y)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average detection confidence"""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

class ByteTracker:
    """ByteTrack implementation for multi-object tracking"""
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: float = 10
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections"""
        
        self.frame_count += 1
        
        # Split detections by confidence
        high_conf_dets = []
        low_conf_dets = []
        
        for det in detections:
            if self._get_box_area(det.bbox) > self.min_box_area:
                if det.confidence >= self.track_thresh:
                    high_conf_dets.append(det)
                else:
                    low_conf_dets.append(det)
        
        # Get active tracks
        active_tracks = [t for t in self.tracks.values() if t.is_active]
        
        # First association with high confidence detections
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            active_tracks, high_conf_dets
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = active_tracks[track_idx]
            detection = high_conf_dets[det_idx]
            track.update(detection, self.frame_count)
        
        # Second association with low confidence detections
        if unmatched_tracks and low_conf_dets:
            remaining_tracks = [active_tracks[i] for i in unmatched_tracks]
            matched_tracks2, unmatched_tracks2, unmatched_dets2 = self._associate(
                remaining_tracks, low_conf_dets, thresh=0.5
            )
            
            # Update matched tracks
            for track_idx, det_idx in matched_tracks2:
                track = remaining_tracks[track_idx]
                detection = low_conf_dets[det_idx]
                track.update(detection, self.frame_count)
            
            # Update unmatched lists
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks2]
            unmatched_dets = [high_conf_dets[i] for i in unmatched_dets]
            unmatched_dets.extend([low_conf_dets[i] for i in unmatched_dets2])
        else:
            unmatched_dets = [high_conf_dets[i] for i in unmatched_dets]
        
        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            if det.confidence >= self.track_thresh:
                new_track = Track(track_id=self.next_id)
                new_track.update(det, self.frame_count)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            track = active_tracks[track_idx]
            track.frames_since_update += 1
            
            # Deactivate lost tracks
            if track.frames_since_update > self.track_buffer:
                track.is_active = False
        
        # Return active tracks
        return [t for t in self.tracks.values() if t.is_active]
    
    def _associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
        thresh: Optional[float] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU"""
        
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        thresh = thresh or self.match_thresh
        
        # Build cost matrix (IoU distances)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_box = self._predict_box(track)
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track_box, det.bbox)
                cost_matrix[i, j] = 1 - iou
        
        # Solve assignment problem
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        
        matches = []
        unmatched_tracks = []
        unmatched_dets = []
        
        for i in range(len(tracks)):
            if x[i] >= 0:
                matches.append((i, x[i]))
            else:
                unmatched_tracks.append(i)
        
        for j in range(len(detections)):
            if y[j] < 0:
                unmatched_dets.append(j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _predict_box(self, track: Track) -> Tuple[float, float, float, float]:
        """Predict bounding box for track"""
        if not track.detections:
            return (0, 0, 0, 0)
        
        last_box = track.detections[-1].bbox
        
        if len(track.detections) < 2:
            return last_box
        
        # Simple motion model
        prev_box = track.detections[-2].bbox
        
        dx = last_box[0] - prev_box[0]
        dy = last_box[1] - prev_box[1]
        
        predicted_box = (
            last_box[0] + dx,
            last_box[1] + dy,
            last_box[2] + dx,
            last_box[3] + dy
        )
        
        return predicted_box
    
    def _calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate IoU between two boxes"""
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _get_box_area(self, box: Tuple[float, float, float, float]) -> float:
        """Calculate box area"""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def process_video(self, detections_per_frame: List[List[Detection]]) -> List[Track]:
        """Process entire video and return all tracks"""
        
        logger.info(f"Processing {len(detections_per_frame)} frames")
        
        for frame_idx, frame_detections in enumerate(detections_per_frame):
            self.update(frame_detections)
        
        # Get all tracks (including inactive)
        all_tracks = list(self.tracks.values())
        
        logger.info(f"Generated {len(all_tracks)} tracks")
        
        return all_tracks
    
    def get_track_statistics(self) -> Dict:
        """Get tracking statistics"""
        
        active_tracks = [t for t in self.tracks.values() if t.is_active]
        completed_tracks = [t for t in self.tracks.values() if not t.is_active]
        
        track_lengths = [len(t.frames) for t in self.tracks.values()]
        
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active_tracks),
            "completed_tracks": len(completed_tracks),
            "avg_track_length": np.mean(track_lengths) if track_lengths else 0,
            "max_track_length": max(track_lengths) if track_lengths else 0,
            "min_track_length": min(track_lengths) if track_lengths else 0,
            "total_frames_processed": self.frame_count
        }