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
import torch.nn as nn
import torchvision.models as models
import logging
from enum import Enum
import json

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


class FeatureExtractor(nn.Module):
    """Extracteur de features visuelles pour ré-identification"""
    
    def __init__(self, feature_dim=256):
        super().__init__()
        # Utiliser ResNet50 pré-entraîné
        resnet = models.resnet50(pretrained=True)
        # Retirer la dernière couche
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Ajouter projection vers dimension voulue
        self.projection = nn.Linear(2048, feature_dim)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """Extraire features d'une image"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        # Normaliser pour distance cosinus
        features = nn.functional.normalize(features, p=2, dim=1)
        return features


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
            'use_cuda': torch.cuda.is_available(),
        }
    
    def _init_feature_extractor(self):
        """Initialiser extracteur de features visuelles"""
        if not self.config['use_appearance']:
            return None
            
        try:
            extractor = FeatureExtractor(self.config['feature_dim'])
            if self.config['use_cuda']:
                extractor = extractor.cuda()
            extractor.eval()
            logger.info("Feature extractor initialisé")
            return extractor
        except Exception as e:
            logger.warning(f"Impossible d'initialiser feature extractor: {e}")
            return None
    
    def update(self, detections: List[Detection]) -> List[STrack]:
        """
        Mettre à jour tracks avec nouvelles détections
        
        Args:
            detections: Liste des détections courantes
            
        Returns:
            Liste des tracks actifs
        """
        self.frame_id += 1
        
        # Séparer détections par score
        high_det = []
        low_det = []
        
        for det in detections:
            if det.score > self.det_thresh:
                high_det.append(det)
            elif det.score > self.track_thresh:
                low_det.append(det)
        
        # Tracks à différents états
        activated_stracks = []
        refind_stracks = []
        
        # Association des détections haute confiance
        if len(high_det) > 0:
            # Prédire positions des tracks existants
            for track in self.tracked_stracks:
                track.predict()
            
            # Calculer matrice de coût
            dists = self._compute_distance_matrix(
                self.tracked_stracks, 
                high_det,
                use_appearance=self.config['use_appearance']
            )
            
            # Association Hungarian algorithm
            matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
                dists, 
                thresh=self.match_thresh
            )
            
            # Update matched tracks
            for t_idx, d_idx in matches:
                track = self.tracked_stracks[t_idx]
                det = high_det[d_idx]
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            
            # Créer nouveaux tracks pour détections non matchées
            for d_idx in unmatched_dets:
                track = STrack(high_det[d_idx], self.frame_id)
                activated_stracks.append(track)
            
            # Gérer tracks non matchés
            for t_idx in unmatched_tracks:
                track = self.tracked_stracks[t_idx]
                track.mark_lost()
                self.lost_stracks.append(track)
        
        # Essayer de récupérer tracks perdus avec détections basse confiance
        if len(self.lost_stracks) > 0 and len(low_det) > 0:
            dists = self._compute_distance_matrix(
                self.lost_stracks,
                low_det,
                use_appearance=True  # Toujours utiliser appearance pour lost tracks
            )
            
            matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
                dists,
                thresh=0.5  # Seuil plus strict pour ré-identification
            )
            
            for t_idx, d_idx in matches:
                track = self.lost_stracks[t_idx]
                det = low_det[d_idx]
                track.update(det, self.frame_id)
                refind_stracks.append(track)
        
        # Mettre à jour listes de tracks
        self.tracked_stracks = self._merge_track_lists(
            activated_stracks, 
            refind_stracks
        )
        
        # Supprimer tracks perdus trop longtemps
        self.lost_stracks = [
            t for t in self.lost_stracks 
            if t.time_since_update < self.max_time_lost
        ]
        
        # Filtrer tracks par durée minimale
        output_stracks = [
            t for t in self.tracked_stracks
            if t.tracklet_len >= 3
        ]
        
        # Calculer métriques
        self._update_metrics(output_stracks)
        
        return output_stracks
    
    def _compute_distance_matrix(
        self, 
        tracks: List[STrack], 
        detections: List[Detection],
        use_appearance: bool = True
    ) -> np.ndarray:
        """
        Calculer matrice de distance entre tracks et détections
        Combine IoU et similarité visuelle
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 0))
        
        # Extraire bboxes
        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = np.array([d.bbox for d in detections])
        
        # Calculer IoU distances
        iou_dists = 1 - self._compute_iou_matrix(track_bboxes, det_bboxes)
        
        if not use_appearance or self.feature_extractor is None:
            return iou_dists
        
        # Calculer distances visuelles
        visual_dists = self._compute_visual_distances(tracks, detections)
        
        # Combiner distances (70% IoU, 30% visual)
        combined_dists = 0.7 * iou_dists + 0.3 * visual_dists
        
        # Appliquer contraintes football
        combined_dists = self._apply_football_constraints(
            combined_dists, tracks, detections
        )
        
        return combined_dists
    
    def _compute_iou_matrix(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """Calculer matrice IoU entre deux ensembles de bboxes"""
        # Utiliser cache si possible
        cache_key = (bboxes1.tobytes(), bboxes2.tobytes())
        if cache_key in self.iou_cache:
            return self.iou_cache[cache_key]
        
        # Calcul vectorisé IoU
        x1_1, y1_1, x2_1, y2_1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
        x1_2, y1_2, x2_2, y2_2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]
        
        # Aires
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Intersections
        xx1 = np.maximum(x1_1[:, None], x1_2)
        yy1 = np.maximum(y1_1[:, None], y1_2)
        xx2 = np.minimum(x2_1[:, None], x2_2)
        yy2 = np.minimum(y2_1[:, None], y2_2)
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        union = area1[:, None] + area2 - inter
        
        iou = inter / (union + 1e-6)
        
        # Mettre en cache
        self.iou_cache[cache_key] = iou
        
        return iou
    
    def _compute_visual_distances(
        self, 
        tracks: List[STrack], 
        detections: List[Detection]
    ) -> np.ndarray:
        """Calculer distances visuelles entre tracks et détections"""
        n_tracks = len(tracks)
        n_dets = len(detections)
        dists = np.ones((n_tracks, n_dets))
        
        for i, track in enumerate(tracks):
            track_feat = track.get_feature()
            if track_feat is None:
                continue
                
            for j, det in enumerate(detections):
                if det.feature is None:
                    continue
                
                # Distance cosinus
                dist = 1 - np.dot(track_feat, det.feature) / (
                    np.linalg.norm(track_feat) * np.linalg.norm(det.feature) + 1e-6
                )
                dists[i, j] = dist
        
        return dists
    
    def _apply_football_constraints(
        self,
        dists: np.ndarray,
        tracks: List[STrack],
        detections: List[Detection]
    ) -> np.ndarray:
        """Appliquer contraintes spécifiques au football"""
        modified_dists = dists.copy()
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Pénaliser changement d'équipe
                if track.team_id is not None and det.team_id is not None:
                    if track.team_id != det.team_id:
                        modified_dists[i, j] += 0.5
                
                # Favoriser cohérence couleur maillot
                if track.jersey_color is not None and det.jersey_color is not None:
                    color_dist = np.linalg.norm(track.jersey_color - det.jersey_color)
                    if color_dist > 50:  # Seuil couleur
                        modified_dists[i, j] += 0.3
                
                # Contrainte de distance maximale (un joueur ne peut pas téléporter)
                if len(track.position_history) > 0:
                    last_pos = track.position_history[-1]
                    det_center = track._bbox_to_center(det.bbox)[:2]
                    spatial_dist = np.linalg.norm(last_pos - det_center)
                    
                    max_dist = 50  # pixels max entre frames
                    if spatial_dist > max_dist:
                        modified_dists[i, j] += 1.0
        
        return modified_dists
    
    def _linear_assignment(
        self, 
        dists: np.ndarray, 
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Résoudre assignment avec Hungarian algorithm
        
        Returns:
            matches: Liste de (track_idx, det_idx)
            unmatched_tracks: Indices tracks non matchés
            unmatched_dets: Indices détections non matchées
        """
        if dists.size == 0:
            return [], list(range(dists.shape[0])), list(range(dists.shape[1]))
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(dists)
        
        matches = []
        unmatched_tracks = list(range(dists.shape[0]))
        unmatched_dets = list(range(dists.shape[1]))
        
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] < thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_dets.remove(c)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _merge_track_lists(self, *track_lists) -> List[STrack]:
        """Fusionner listes de tracks en évitant doublons"""
        merged = []
        track_ids = set()
        
        for tracks in track_lists:
            for track in tracks:
                if track.track_id not in track_ids:
                    merged.append(track)
                    track_ids.add(track.track_id)
        
        return merged
    
    def _update_metrics(self, tracks: List[STrack]):
        """Mettre à jour métriques de tracking"""
        # TODO: Implémenter calcul MOTA, MOTP, IDF1
        pass
    
    def get_metrics(self) -> TrackMetrics:
        """Obtenir métriques de tracking"""
        return self.metrics
    
    def visualize_tracks(
        self, 
        image: np.ndarray, 
        tracks: List[STrack],
        show_trajectory: bool = True,
        show_id: bool = True,
        show_team: bool = True
    ) -> np.ndarray:
        """
        Visualiser tracks sur image
        
        Args:
            image: Image à annoter
            tracks: Tracks à visualiser
            show_trajectory: Afficher trajectoire
            show_id: Afficher ID
            show_team: Afficher équipe
            
        Returns:
            Image annotée
        """
        img_vis = image.copy()
        
        # Couleurs par équipe
        team_colors = {
            0: (255, 0, 0),    # Équipe 1 - Rouge
            1: (0, 0, 255),    # Équipe 2 - Bleu
            2: (0, 255, 0),    # Arbitres - Vert
            None: (255, 255, 0) # Non assigné - Jaune
        }
        
        for track in tracks:
            # Couleur selon équipe
            color = team_colors.get(track.team_id, (255, 255, 0))
            
            # Dessiner bbox
            x1, y1, x2, y2 = track.bbox.astype(int)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Afficher ID
            if show_id:
                text = f"ID:{track.track_id}"
                if show_team and track.team_id is not None:
                    text += f" T:{track.team_id}"
                
                cv2.putText(
                    img_vis, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
            
            # Afficher trajectoire
            if show_trajectory and len(track.position_history) > 1:
                points = np.array(list(track.position_history), dtype=np.int32)
                cv2.polylines(img_vis, [points], False, color, 2)
        
        return img_vis
    
    def export_tracks(
        self, 
        format: str = 'mot',
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Exporter tracks dans format standard
        
        Args:
            format: Format d'export ('mot', 'json', 'csv')
            output_path: Chemin de sortie
            
        Returns:
            Données exportées
        """
        all_tracks = self.tracked_stracks + self.lost_stracks
        
        if format == 'mot':
            # Format MOT Challenge
            data = []
            for track in all_tracks:
                for frame in range(track.start_frame, track.frame_id + 1):
                    if frame in track.track_history:
                        bbox = track.track_history[frame]
                        data.append([
                            frame,
                            track.track_id,
                            bbox[0], bbox[1],
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            track.score,
                            track.class_id,
                            1  # visibility
                        ])
            
            if output_path:
                np.savetxt(output_path, data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d')
            
            return {'format': 'mot', 'data': data}
        
        elif format == 'json':
            # Format JSON détaillé
            data = {
                'tracks': [],
                'metadata': {
                    'total_frames': self.frame_id,
                    'total_tracks': len(all_tracks),
                    'metrics': self.metrics.__dict__
                }
            }
            
            for track in all_tracks:
                track_data = {
                    'track_id': track.track_id,
                    'team_id': track.team_id,
                    'class_id': track.class_id,
                    'frames': track.track_history,
                    'stats': {
                        'distance_covered': track.distance_covered,
                        'max_speed': track.max_speed,
                        'avg_speed': track.avg_speed
                    }
                }
                data['tracks'].append(track_data)
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            return data
        
        else:
            raise ValueError(f"Format non supporté: {format}")


# Fonctions utilitaires
def evaluate_tracking(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> TrackMetrics:
    """
    Évaluer performance du tracking
    
    Args:
        predictions: Prédictions du tracker
        ground_truth: Vérité terrain
        
    Returns:
        Métriques de tracking
    """
    metrics = TrackMetrics()
    
    # Implémenter calcul MOTA
    # MOTA = 1 - (FN + FP + IDSW) / GT
    # où FN = False Negatives, FP = False Positives, IDSW = ID Switches, GT = Ground Truth
    
    # Implémenter calcul MOTP
    # MOTP = Σ(d_i) / Σ(c_i)
    # où d_i = distance entre prédiction et GT, c_i = nombre de correspondances
    
    # Implémenter calcul IDF1
    # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    
    return metrics


if __name__ == "__main__":
    # Test du tracker
    tracker = ByteTracker()
    
    # Créer détections de test
    detections = [
        Detection(
            bbox=np.array([100, 100, 150, 200]),
            score=0.9,
            class_id=ObjectClass.PLAYER.value,
            team_id=0
        ),
        Detection(
            bbox=np.array([200, 150, 250, 250]),
            score=0.85,
            class_id=ObjectClass.PLAYER.value,
            team_id=1
        )
    ]
    
    # Update tracker
    tracks = tracker.update(detections)
    
    logger.info(f"Nombre de tracks: {len(tracks)}")
    for track in tracks:
        logger.info(f"Track {track.track_id}: Team {track.team_id}, State {track.state}")