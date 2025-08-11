"""
ByteTrack implementation for football player tracking
Adapté pour gérer 22+ joueurs avec ré-identification robuste
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import cv2
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import torch
import logging
from enum import Enum

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class TrackState(Enum):
    """États possibles d'un track"""
    NEW = 0          # Nouveau track
    TRACKED = 1      # Track confirmé et suivi
    LOST = 2         # Track perdu temporairement
    REMOVED = 3      # Track supprimé


class ObjectClass(Enum):
    """Classes d'objets pour le tracking"""
    PLAYER = 0
    REFEREE = 1
    BALL = 2
    GOALKEEPER = 3


@dataclass
class Detection:
    """Structure pour une détection"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    score: float              # Confidence score
    class_id: int             # Class ID
    feature: Optional[np.ndarray] = None  # Visual feature vector
    team_id: Optional[int] = None         # Team identification
    jersey_color: Optional[np.ndarray] = None  # Couleur dominante du maillot


@dataclass
class TrackMetrics:
    """Métriques de tracking"""
    mota: float = 0.0    # Multiple Object Tracking Accuracy
    motp: float = 0.0    # Multiple Object Tracking Precision
    idf1: float = 0.0    # ID F1 Score
    num_switches: int = 0
    num_false_positives: int = 0
    num_misses: int = 0
    precision_by_team: Dict[int, float] = field(default_factory=dict)


class STrack:
    """Single Track object pour ByteTrack"""
    
    shared_kalman = KalmanFilter(dim_x=7, dim_z=4)
    _count = 0
    
    def __init__(self, detection: Detection, frame_id: int):
        self.track_id = STrack._count
        STrack._count += 1
        
        # Information de base
        self.detection = detection
        self.bbox = detection.bbox
        self.score = detection.score
        self.class_id = detection.class_id
        self.state = TrackState.NEW
        
        # Information football spécifique
        self.team_id = detection.team_id
        self.jersey_color = detection.jersey_color
        self.jersey_number = None
        
        # Tracking temporel
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.time_since_update = 0
        
        # Features visuelles pour ré-identification
        self.features = deque(maxlen=30)  # Garder 30 dernières features
        if detection.feature is not None:
            self.features.append(detection.feature)
        
        # Historique positions pour analyse tactique
        self.position_history = deque(maxlen=300)  # ~10 secondes à 30fps
        self.velocity_history = deque(maxlen=30)
        
        # Kalman filter pour prédiction mouvement
        self.kalman_filter = None
        self._init_kalman()
        
        # Statistiques du joueur
        self.distance_covered = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.time_in_possession = 0.0
    
    def _init_kalman(self):
        """Initialiser le filtre de Kalman pour prédiction position"""
        self.kalman_filter = KalmanFilter(dim_x=7, dim_z=4)
        
        # État: [x_center, y_center, width, height, vx, vy, area_change_rate]
        # Mesure: [x_center, y_center, width, height]
        
        # Matrice de transition
        self.kalman_filter.F = np.eye(7)
        self.kalman_filter.F[0, 4] = 1  # x += vx
        self.kalman_filter.F[1, 5] = 1  # y += vy
        
        # Matrice de mesure
        self.kalman_filter.H = np.zeros((4, 7))
        self.kalman_filter.H[:4, :4] = np.eye(4)
        
        # Bruit process et mesure
        self.kalman_filter.R *= 10  # Bruit de mesure
        self.kalman_filter.Q[4:, 4:] *= 0.01  # Bruit de process pour vitesse
        self.kalman_filter.P *= 100  # Incertitude initiale
        
        # Initialisation avec détection
        bbox_center = self._bbox_to_center(self.bbox)
        self.kalman_filter.x[:4] = bbox_center
        
    def _bbox_to_center(self, bbox):
        """Convertir bbox en format [cx, cy, w, h]"""
        x1, y1, x2, y2 = bbox
        return np.array([
            (x1 + x2) / 2,  # center x
            (y1 + y2) / 2,  # center y
            x2 - x1,        # width
            y2 - y1         # height
        ])
    
    def _center_to_bbox(self, center):
        """Convertir [cx, cy, w, h] en bbox [x1, y1, x2, y2]"""
        cx, cy, w, h = center
        return np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ])
    
    def predict(self):
        """Prédire la position suivante avec Kalman filter"""
        self.kalman_filter.predict()
        self.bbox = self._center_to_bbox(self.kalman_filter.x[:4])
        
    def update(self, detection: Detection, frame_id: int):
        """Mettre à jour le track avec nouvelle détection"""
        self.detection = detection
        self.score = detection.score
        self.tracklet_len += 1
        self.time_since_update = 0
        self.frame_id = frame_id
        
        # Update Kalman filter
        bbox_center = self._bbox_to_center(detection.bbox)
        self.kalman_filter.update(bbox_center)
        self.bbox = detection.bbox
        
        # Update features visuelles
        if detection.feature is not None:
            self.features.append(detection.feature)
        
        # Update position history
        center = bbox_center[:2]
        self.position_history.append(center)
        
        # Calculer vitesse
        if len(self.position_history) > 1:
            velocity = self.position_history[-1] - self.position_history[-2]
            self.velocity_history.append(velocity)
            
            # Update stats
            speed = np.linalg.norm(velocity)
            self.max_speed = max(self.max_speed, speed)
        
        # Update état
        if self.state == TrackState.NEW and self.tracklet_len >= 3:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED
            
    def mark_lost(self):
        """Marquer le track comme perdu"""
        self.state = TrackState.LOST
        self.time_since_update += 1
        
    def mark_removed(self):
        """Marquer le track comme supprimé"""
        self.state = TrackState.REMOVED
    
    def get_feature(self):
        """Obtenir la feature moyenne pour ré-identification"""
        if len(self.features) == 0:
            return None
        return np.mean(list(self.features), axis=0)

class ByteTracker:
    """
    ByteTracker adapté pour le football
    Gestion de 22+ joueurs avec ré-identification robuste
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialiser ByteTracker
        
        Args:
            config: Configuration du tracker
        """
        self.config = config or self._get_default_config()
        
        # Tracks actifs
        self.tracked_stracks: List[STrack] = []      # Tracks confirmés
        self.lost_stracks: List[STrack] = []         # Tracks perdus
        self.removed_stracks: List[STrack] = []      # Tracks supprimés
        
        # Frame counter
        self.frame_id = 0
        
        # Paramètres de tracking
        self.det_thresh = self.config['det_thresh']
        self.match_thresh = self.config['match_thresh']
        self.track_thresh = self.config['track_thresh']
        self.max_time_lost = self.config['max_time_lost']
        
        # Features extractor pour ré-identification
        self.feature_extractor = self._init_feature_extractor()
        
        # Métriques
        self.metrics = TrackMetrics()
        self.track_history = defaultdict(list)
        
        # Cache pour optimisation
        self.iou_cache = {}
        self.feature_cache = {}
        
        logger.info("ByteTracker initialisé pour tracking football")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour football"""
        return {
            'det_thresh': 0.6,        # Seuil détection
            'match_thresh': 0.8,      # Seuil matching IoU
            'track_thresh': 0.5,      # Seuil track confirmation
            'max_time_lost': 30,      # Frames max avant suppression (1 sec à 30fps)
            'min_box_area': 100,      # Aire minimale bbox
            'feature_dim': 256,       # Dimension features visuelles
            'use_appearance': True,   # Utiliser features visuelles
            'max_players': 30,        # Max joueurs simultanés
            'team_consistency': True,  # Forcer cohérence équipe
        }
    
    def _init_feature_extractor(self):
        """Initialiser extracteur de features visuelles"""
        # TODO: Implémenter ResNet ou autre pour extraction features
        # Pour l'instant, retourner None
        return None
    
    def update(self, detections: List[Detection]) -> List[STrack]:
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
        tracks: List[STrack],
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
    
    def _predict_box(self, track: STrack) -> Tuple[float, float, float, float]:
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
    
    def process_video(self, detections_per_frame: List[List[Detection]]) -> List[STrack]:
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