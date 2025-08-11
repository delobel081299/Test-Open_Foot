"""
Pose Extractor 3D pour analyse biomécanique football
Extraction optimisée avec MediaPipe et post-processing avancé
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import logging

# Try to import mediapipe, but handle if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# Try to import scipy for filtering
try:
    from scipy.signal import savgol_filter
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

from backend.core.tracking.byte_tracker import STrack
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Pose3D:
    """Structure pour pose 3D avec métadonnées complètes"""
    keypoints: np.ndarray  # 33 keypoints x 4 (x, y, z, visibility)
    world_landmarks: Optional[np.ndarray] = None  # 33 x 3 world coordinates
    confidence: float = 0.0
    track_id: int = -1
    frame_number: int = -1
    joint_angles: Dict[str, float] = field(default_factory=dict)
    center_of_mass: Optional[np.ndarray] = None  # 3D COM
    body_orientation: Optional[np.ndarray] = None  # Euler angles
    segmentation_mask: Optional[np.ndarray] = None
    interpolated_points: List[int] = field(default_factory=list)  # IDs des points interpolés
    
    def get_keypoint(self, landmark_id: int) -> Tuple[float, float, float, float]:
        """Get specific keypoint (x, y, z, visibility)"""
        return tuple(self.keypoints[landmark_id])
    
    def get_world_landmark(self, landmark_id: int) -> Optional[Tuple[float, float, float]]:
        """Get world coordinates for landmark"""
        if self.world_landmarks is not None:
            return tuple(self.world_landmarks[landmark_id])
        return None
    
    def is_visible(self, landmark_id: int, threshold: float = 0.5) -> bool:
        """Check if keypoint is visible"""
        return self.keypoints[landmark_id][3] > threshold
    
    def was_interpolated(self, landmark_id: int) -> bool:
        """Check if keypoint was interpolated"""
        return landmark_id in self.interpolated_points


@dataclass
class PoseExtractionConfig:
    """Configuration pour l'extraction de pose"""
    model_complexity: int = 2  # 0, 1, or 2 (heavy)
    enable_segmentation: bool = True
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Optimisation
    batch_size: int = 8
    use_gpu: bool = True
    max_workers: int = 4
    
    # Preprocessing
    crop_padding: float = 0.2  # 20% padding autour du joueur
    target_size: Tuple[int, int] = (368, 368)  # Taille optimale pour MediaPipe
    
    # Post-processing
    enable_temporal_smoothing: bool = True
    smoothing_window: int = 5  # Frames pour Savitzky-Golay
    interpolation_max_gap: int = 3  # Max frames pour interpolation
    visibility_threshold: float = 0.3
    
    # Normalisation
    reference_height: float = 1.75  # Hauteur référence en mètres
    normalize_by_height: bool = True


class TemporalSmoother:
    """Lissage temporel des keypoints avec Savitzky-Golay"""
    
    def __init__(self, window_length: int = 5, polyorder: int = 2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.buffers: Dict[int, deque] = {}  # Buffer par landmark
        
    def add_frame(self, keypoints: np.ndarray, track_id: int):
        """Ajouter frame au buffer"""
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.window_length * 2)
        
        self.buffers[track_id].append(keypoints.copy())
    
    def smooth_keypoints(self, keypoints: np.ndarray, track_id: int) -> np.ndarray:
        """Appliquer lissage si assez de frames"""
        if not SCIPY_AVAILABLE or track_id not in self.buffers:
            return keypoints
        
        buffer = self.buffers[track_id]
        if len(buffer) < self.window_length:
            return keypoints
        
        # Stack frames pour lissage
        frames = np.array(list(buffer))
        smoothed = keypoints.copy()
        
        # Lisser chaque coordonnée séparément
        for landmark_id in range(33):
            for coord_idx in range(3):  # x, y, z
                try:
                    values = frames[:, landmark_id, coord_idx]
                    if len(values) >= self.window_length:
                        smoothed[landmark_id, coord_idx] = savgol_filter(
                            values, 
                            min(self.window_length, len(values)), 
                            min(self.polyorder, len(values) - 1)
                        )[-1]
                except:
                    pass  # Garder valeur originale si erreur
        
        return smoothed

class PoseExtractor:
    """Extracteur de pose 3D optimisé pour analyse biomécanique football"""
    
    def __init__(self, config: Optional[PoseExtractionConfig] = None):
        """
        Initialiser l'extracteur de pose
        
        Args:
            config: Configuration d'extraction
        """
        self.config = config or PoseExtractionConfig()
        
        # Initialiser MediaPipe
        self.mp_pose = None
        self.mp_drawing = None
        self.pose_detector = None
        self._init_mediapipe()
        
        # Lissage temporel
        self.temporal_smoother = TemporalSmoother(
            window_length=self.config.smoothing_window,
            polyorder=min(2, self.config.smoothing_window - 1)
        )
        
        # Key landmark indices pour analyse football
        self.key_landmarks = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # Cache pour optimisation
        self.pose_cache = {}
        self.batch_processing = self.config.batch_size > 1
        
        # Statistiques de performance
        self.processing_times = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        logger.info(f"PoseExtractor initialisé avec modèle complexité {self.config.model_complexity}")
        if TORCH_AVAILABLE and self.config.use_gpu:
            logger.info("GPU détecté et activé pour accélération")
    
    def _init_mediapipe(self):
        """Initialiser MediaPipe Pose"""
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe non disponible. Extraction impossible.")
            return
        
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config.model_complexity,
                enable_segmentation=self.config.enable_segmentation,
                smooth_landmarks=self.config.smooth_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            
            logger.info("MediaPipe Pose initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation MediaPipe: {e}")
            self.pose_detector = None
    
    def _preprocess_crop(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocessing intelligent du crop joueur
        
        Args:
            image: Image complète
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            (cropped_image, metadata)
        """
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = image.shape[:2]
        
        # Calculer padding
        bbox_w, bbox_h = x2 - x1, y2 - y1
        pad_w = int(bbox_w * self.config.crop_padding)  
        pad_h = int(bbox_h * self.config.crop_padding)
        
        # Ajuster coordonnées avec padding
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)
        
        # Extraire crop avec padding
        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size == 0:
            return image, {'error': 'Invalid crop'}
        
        # Redimensionner à taille cible
        crop_resized = cv2.resize(crop, self.config.target_size)
        
        # Métadonnées pour reproject coordinates
        metadata = {
            'original_bbox': bbox,
            'padded_bbox': np.array([x1_pad, y1_pad, x2_pad, y2_pad]),
            'scale_x': (x2_pad - x1_pad) / self.config.target_size[0],
            'scale_y': (y2_pad - y1_pad) / self.config.target_size[1],
            'offset_x': x1_pad,
            'offset_y': y1_pad
        }
        
        return crop_resized, metadata
    
    def extract_pose(self, image: np.ndarray, track_id: int, frame_number: int, 
                    bbox: Optional[np.ndarray] = None) -> Optional[Pose3D]:
        """
        Extraire pose 3D d'une image avec preprocessing avancé
        
        Args:
            image: Image ou crop du joueur
            track_id: ID du track
            frame_number: Numéro de frame
            bbox: Bounding box optionnelle pour crop intelligent
            
        Returns:
            Pose3D ou None si échec
        """
        start_time = time.time()
        
        if not MEDIAPIPE_AVAILABLE or self.pose_detector is None:
            return self._create_fallback_pose(track_id, frame_number)
        
        try:
            # Preprocessing du crop si bbox fournie
            if bbox is not None:
                processed_image, metadata = self._preprocess_crop(image, bbox)
            else:
                processed_image = cv2.resize(image, self.config.target_size)
                metadata = {'scale_x': 1.0, 'scale_y': 1.0, 'offset_x': 0, 'offset_y': 0}
            
            # Convertir BGR vers RGB
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Extraction MediaPipe
            results = self.pose_detector.process(rgb_image)
            
            if not results.pose_landmarks:
                self.success_rate.append(0)
                return None
            
            # Extraire keypoints 2D/3D
            keypoints = self._extract_keypoints(results.pose_landmarks, metadata)
            
            # Extraire world landmarks si disponibles
            world_landmarks = None
            if results.pose_world_landmarks:
                world_landmarks = self._extract_world_landmarks(results.pose_world_landmarks)
            
            # Extraire masque de segmentation
            segmentation_mask = None
            if self.config.enable_segmentation and results.segmentation_mask is not None:
                segmentation_mask = results.segmentation_mask
            
            # Calculer confiance globale
            confidence = self._calculate_confidence(keypoints)
            
            # Créer objet Pose3D
            pose = Pose3D(
                keypoints=keypoints,
                world_landmarks=world_landmarks,
                confidence=confidence,
                track_id=track_id,
                frame_number=frame_number,
                segmentation_mask=segmentation_mask
            )
            
            # Post-processing
            pose = self._postprocess_pose(pose)
            
            # Mesurer performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.success_rate.append(1)
            
            return pose
            
        except Exception as e:
            logger.error(f"Erreur extraction pose track {track_id}: {e}")
            self.success_rate.append(0)
            return self._create_fallback_pose(track_id, frame_number)
    
    def _extract_keypoints(self, landmarks, metadata: Dict[str, Any]) -> np.ndarray:
        """Extraire et reprojeter keypoints"""
        keypoints = np.zeros((33, 4))  # x, y, z, visibility
        
        for i, landmark in enumerate(landmarks.landmark):
            # Coordonnées normalisées -> pixels dans crop
            x_crop = landmark.x * self.config.target_size[0]
            y_crop = landmark.y * self.config.target_size[1]
            
            # Reprojection vers coordonnées image originale
            x_orig = x_crop * metadata['scale_x'] + metadata['offset_x']
            y_orig = y_crop * metadata['scale_y'] + metadata['offset_y']
            
            keypoints[i] = [x_orig, y_orig, landmark.z, landmark.visibility]
        
        return keypoints
    
    def _extract_world_landmarks(self, world_landmarks) -> np.ndarray:
        """Extraire coordonnées monde 3D"""
        world_coords = np.zeros((33, 3))
        
        for i, landmark in enumerate(world_landmarks.landmark):
            world_coords[i] = [landmark.x, landmark.y, landmark.z]
        
        return world_coords
    
    def _calculate_confidence(self, keypoints: np.ndarray) -> float:
        """Calculer confiance globale de la pose"""
        # Visibility moyenne des points clés
        key_indices = [11, 12, 23, 24, 25, 26, 27, 28]  # Torse et jambes
        key_confidences = keypoints[key_indices, 3]
        
        return float(np.mean(key_confidences[key_confidences > 0]))
    
    def _postprocess_pose(self, pose: Pose3D) -> Pose3D:
        """Post-processing avancé de la pose"""
        
        # 1. Filtrer points occludés
        pose.keypoints = self._filter_occluded_points(pose.keypoints)
        
        # 2. Interpoler points manquants
        pose.keypoints, interpolated = self._interpolate_missing_points(pose.keypoints)
        pose.interpolated_points = interpolated
        
        # 3. Lissage temporel
        if self.config.enable_temporal_smoothing:
            self.temporal_smoother.add_frame(pose.keypoints, pose.track_id)
            pose.keypoints = self.temporal_smoother.smooth_keypoints(pose.keypoints, pose.track_id)
        
        # 4. Normalisation par taille
        if self.config.normalize_by_height:
            pose.keypoints = self._normalize_by_height(pose.keypoints)
        
        # 5. Calculer angles articulaires
        pose.joint_angles = self.calculate_joint_angles(pose)
        
        # 6. Calculer centre de masse
        pose.center_of_mass = self._calculate_center_of_mass(pose)
        
        # 7. Calculer orientation corps
        pose.body_orientation = self._calculate_body_orientation(pose)
        
        return pose
    
    def _filter_occluded_points(self, keypoints: np.ndarray) -> np.ndarray:
        """Filtrer points occludés basé sur visibility"""
        filtered = keypoints.copy()
        
        # Masquer points avec faible visibility
        low_vis_mask = keypoints[:, 3] < self.config.visibility_threshold
        filtered[low_vis_mask, 3] = 0.0
        
        return filtered
    
    def _interpolate_missing_points(self, keypoints: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Interpoler points manquants via symétrie corporelle"""
        interpolated = keypoints.copy()
        interpolated_ids = []
        
        # Paires symétriques pour interpolation
        symmetric_pairs = [
            (11, 12),  # épaules
            (13, 14),  # coudes  
            (15, 16),  # poignets
            (23, 24),  # hanches
            (25, 26),  # genoux
            (27, 28),  # chevilles
        ]
        
        for left_id, right_id in symmetric_pairs:
            left_vis = keypoints[left_id, 3]
            right_vis = keypoints[right_id, 3]
            
            # Interpoler côté manquant par symétrie
            if left_vis < self.config.visibility_threshold < right_vis:
                # Copier côté droit vers gauche avec symétrie
                right_point = keypoints[right_id]
                # Calculer centre corps pour symétrie
                center_x = (keypoints[11, 0] + keypoints[12, 0]) / 2  # Centre épaules
                
                interpolated[left_id, 0] = center_x - (right_point[0] - center_x)
                interpolated[left_id, 1:3] = right_point[1:3]
                interpolated[left_id, 3] = right_vis * 0.7  # Confiance réduite
                interpolated_ids.append(left_id)
                
            elif right_vis < self.config.visibility_threshold < left_vis:
                # Copier côté gauche vers droit avec symétrie
                left_point = keypoints[left_id]
                center_x = (keypoints[11, 0] + keypoints[12, 0]) / 2
                
                interpolated[right_id, 0] = center_x + (center_x - left_point[0])
                interpolated[right_id, 1:3] = left_point[1:3]
                interpolated[right_id, 3] = left_vis * 0.7
                interpolated_ids.append(right_id)
        
        return interpolated, interpolated_ids
    
    def _normalize_by_height(self, keypoints: np.ndarray) -> np.ndarray:
        """Normaliser keypoints par hauteur du joueur"""
        normalized = keypoints.copy()
        
        # Estimer hauteur joueur (tête -> chevilles)
        head_y = min(keypoints[0, 1], keypoints[7, 1], keypoints[8, 1])  # Tête/oreilles
        ankle_y = max(keypoints[27, 1], keypoints[28, 1])  # Chevilles
        
        player_height_pixels = ankle_y - head_y
        if player_height_pixels > 0:
            # Normaliser par hauteur référence
            scale_factor = self.config.reference_height / (player_height_pixels / 100)  # Approximation
            
            # Appliquer normalisation aux coordonnées spatiales
            normalized[:, :3] *= scale_factor
        
        return normalized
    
    def _calculate_center_of_mass(self, pose: Pose3D) -> np.ndarray:
        """Calculer centre de masse approximatif"""
        if pose.world_landmarks is not None:
            # Utiliser coordonnées monde si disponibles
            coords = pose.world_landmarks
        else:
            coords = pose.keypoints[:, :3]
        
        # Poids approximatifs des segments corporels
        weights = np.ones(33) * 0.1  # Poids par défaut
        weights[[11, 12]] = 0.15  # Épaules
        weights[[23, 24]] = 0.15  # Hanches  
        weights[0] = 0.08  # Tête
        
        # Centre de masse pondéré
        visible_mask = pose.keypoints[:, 3] > self.config.visibility_threshold
        valid_coords = coords[visible_mask]
        valid_weights = weights[visible_mask]
        
        if len(valid_coords) > 0:
            com = np.average(valid_coords, axis=0, weights=valid_weights)
            return com
        
        return np.array([0, 0, 0])
    
    def _calculate_body_orientation(self, pose: Pose3D) -> np.ndarray:
        """Calculer orientation du corps (angles d'Euler)"""
        # Vecteur épaules
        left_shoulder = pose.keypoints[11, :3]
        right_shoulder = pose.keypoints[12, :3]
        shoulder_vec = right_shoulder - left_shoulder
        
        # Vecteur torse (épaules -> hanches)
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        left_hip = pose.keypoints[23, :3]
        right_hip = pose.keypoints[24, :3]
        hip_mid = (left_hip + right_hip) / 2
        torso_vec = hip_mid - shoulder_mid
        
        # Calculer angles d'Euler approximatifs
        # Yaw (rotation autour axe Y)
        yaw = np.arctan2(shoulder_vec[0], shoulder_vec[2])
        
        # Pitch (inclinaison avant/arrière)
        pitch = np.arctan2(torso_vec[2], np.sqrt(torso_vec[0]**2 + torso_vec[1]**2))
        
        # Roll (inclinaison latérale)
        roll = np.arctan2(shoulder_vec[1], shoulder_vec[0])
        
        return np.array([yaw, pitch, roll])
    
    def _create_fallback_pose(self, track_id: int, frame_number: int) -> Pose3D:
        """Créer pose de fallback en cas d'échec"""
        keypoints = np.zeros((33, 4))
        keypoints[:, 3] = 0.1  # Très faible confiance
        
        return Pose3D(
            keypoints=keypoints,
            confidence=0.1,
            track_id=track_id,
            frame_number=frame_number
        )
    
    def extract_poses_batch(
        self,
        crops: List[Tuple[np.ndarray, int, int, Optional[np.ndarray]]]
    ) -> List[Optional[Pose3D]]:
        """
        Extraction batch optimisée pour multiple joueurs
        
        Args:
            crops: Liste de (image, track_id, frame_number, bbox)
            
        Returns:
            Liste de poses extraites
        """
        if self.config.max_workers <= 1:
            # Processing séquentiel
            return [self.extract_pose(crop[0], crop[1], crop[2], crop[3]) 
                   for crop in crops]
        
        # Processing parallèle
        poses = [None] * len(crops)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.extract_pose, crop[0], crop[1], crop[2], crop[3]): idx
                for idx, crop in enumerate(crops)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    poses[idx] = future.result()
                except Exception as e:
                    logger.error(f"Erreur extraction batch index {idx}: {e}")
                    poses[idx] = None
        
        return poses
    
    def extract_poses_from_tracks(
        self,
        tracks: List[STrack],
        frames: List[np.ndarray]
    ) -> Dict[int, List[Pose3D]]:
        """Extraire poses pour tous les tracks avec batch processing"""
        
        logger.info(f"Extraction poses pour {len(tracks)} tracks sur {len(frames)} frames")
        start_time = time.time()
        
        poses_by_track = {}
        
        # Préparer batch jobs
        batch_jobs = []
        
        for track in tracks:
            track_poses = []
            
            # Collecter toutes les détections du track
            for detection, frame_num in zip(track.detections, track.frames):
                if frame_num < len(frames):
                    frame = frames[frame_num]
                    bbox = detection.bbox
                    
                    batch_jobs.append((frame, track.track_id, frame_num, bbox))
        
        # Traitement par batch
        if self.batch_processing and len(batch_jobs) > self.config.batch_size:
            all_poses = []
            
            for i in range(0, len(batch_jobs), self.config.batch_size):
                batch = batch_jobs[i:i + self.config.batch_size]
                batch_poses = self.extract_poses_batch(batch)
                all_poses.extend(batch_poses)
        else:
            # Traitement simple si petit nombre
            all_poses = self.extract_poses_batch(batch_jobs)
        
        # Regrouper par track
        pose_idx = 0
        for track in tracks:
            track_poses = []
            
            for _ in range(len(track.detections)):
                if pose_idx < len(all_poses) and all_poses[pose_idx] is not None:
                    track_poses.append(all_poses[pose_idx])
                pose_idx += 1
            
            if track_poses:
                poses_by_track[track.track_id] = track_poses
        
        processing_time = time.time() - start_time
        logger.info(f"Extraction terminée en {processing_time:.1f}s pour {len(poses_by_track)} joueurs")
        
        return poses_by_track
    
    def calculate_joint_angles(self, pose: Pose3D) -> Dict[str, float]:
        """Calculer angles articulaires complets pour analyse biomécanique"""
        
        angles = {}
        
        try:
            # Angles des genoux (3D si disponible)
            if pose.world_landmarks is not None:
                angles['left_knee_3d'] = self._calculate_angle_3d(
                    pose.world_landmarks[self.key_landmarks['left_hip']],
                    pose.world_landmarks[self.key_landmarks['left_knee']],
                    pose.world_landmarks[self.key_landmarks['left_ankle']]
                )
                angles['right_knee_3d'] = self._calculate_angle_3d(
                    pose.world_landmarks[self.key_landmarks['right_hip']],
                    pose.world_landmarks[self.key_landmarks['right_knee']],
                    pose.world_landmarks[self.key_landmarks['right_ankle']]
                )
            
            # Angles 2D pour compatibilité
            angles['left_knee'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['left_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['left_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['left_ankle'])[:2]
            )
            
            angles['right_knee'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['right_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['right_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['right_ankle'])[:2]
            )
            
            # Angles des hanches
            angles['left_hip'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['left_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['left_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['left_knee'])[:2]
            )
            
            angles['right_hip'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['right_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['right_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['right_knee'])[:2]
            )
            
            # Angles des chevilles
            angles['left_ankle'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['left_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['left_ankle'])[:2],
                pose.get_keypoint(self.key_landmarks['left_foot_index'])[:2]
            )
            
            angles['right_ankle'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['right_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['right_ankle'])[:2],
                pose.get_keypoint(self.key_landmarks['right_foot_index'])[:2]
            )
            
            # Angles des épaules
            angles['left_shoulder'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['left_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['left_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['left_elbow'])[:2]
            )
            
            angles['right_shoulder'] = self._calculate_angle_2d(
                pose.get_keypoint(self.key_landmarks['right_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['right_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['right_elbow'])[:2]
            )
            
            # Angles du torse
            angles['spine_lean'] = self._calculate_spine_angle(pose)
            angles['trunk_rotation'] = self._calculate_trunk_rotation(pose)
            
            # Analyse d'équilibre
            angles['balance_score'] = self._calculate_balance_score(pose)
            angles['weight_distribution'] = self._calculate_weight_distribution(pose)
            
            # Angles spécifiques football
            angles['kicking_leg_angle'] = self._calculate_kicking_preparation(pose)
            angles['running_efficiency'] = self._calculate_running_efficiency(pose)
            
        except Exception as e:
            logger.warning(f"Erreur calcul angles articulaires: {e}")
        
        return angles
    
    def _calculate_angle_3d(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """Calculer angle 3D entre trois points"""
        v1 = point1 - point2
        v2 = point3 - point2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return math.degrees(angle)
    
    def _calculate_angle_2d(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """Calculer angle 2D entre trois points"""
        
        # Vecteurs depuis point2
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculer angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return math.degrees(angle)
    
    def _calculate_trunk_rotation(self, pose: Pose3D) -> float:
        """Calculer rotation du tronc"""
        try:
            left_shoulder = pose.keypoints[self.key_landmarks['left_shoulder'], :3]
            right_shoulder = pose.keypoints[self.key_landmarks['right_shoulder'], :3]
            left_hip = pose.keypoints[self.key_landmarks['left_hip'], :3]
            right_hip = pose.keypoints[self.key_landmarks['right_hip'], :3]
            
            # Vecteurs épaules et hanches
            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip
            
            # Angle entre les vecteurs (rotation tronc)
            cos_angle = np.dot(shoulder_vec[:2], hip_vec[:2]) / (
                np.linalg.norm(shoulder_vec[:2]) * np.linalg.norm(hip_vec[:2]) + 1e-8
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            return math.degrees(angle)
        except:
            return 0.0
    
    def _calculate_weight_distribution(self, pose: Pose3D) -> float:
        """Calculer répartition du poids entre les pieds"""
        try:
            left_ankle = pose.keypoints[self.key_landmarks['left_ankle']]
            right_ankle = pose.keypoints[self.key_landmarks['right_ankle']]
            
            # Distance verticale entre chevilles (indicateur d'appui)
            height_diff = abs(left_ankle[1] - right_ankle[1])
            
            # Score de répartition (0 = équilibré, 100 = tout sur un pied)
            return min(100, height_diff * 2)
        except:
            return 50.0
    
    def _calculate_kicking_preparation(self, pose: Pose3D) -> float:
        """Détecter préparation de frappe"""
        try:
            left_hip = pose.keypoints[self.key_landmarks['left_hip'], :3]
            right_hip = pose.keypoints[self.key_landmarks['right_hip'], :3]
            left_knee = pose.keypoints[self.key_landmarks['left_knee'], :3]
            right_knee = pose.keypoints[self.key_landmarks['right_knee'], :3]
            
            # Analyser flexion des genoux
            left_knee_flex = self._calculate_angle_2d(
                left_hip[:2], left_knee[:2], 
                pose.keypoints[self.key_landmarks['left_ankle'], :2]
            )
            right_knee_flex = self._calculate_angle_2d(
                right_hip[:2], right_knee[:2],
                pose.keypoints[self.key_landmarks['right_ankle'], :2]
            )
            
            # Jambe plus fléchie = jambe de frappe potentielle
            kicking_angle = min(left_knee_flex, right_knee_flex)
            
            # Score de préparation (plus l'angle est petit, plus c'est préparé)
            preparation_score = max(0, 180 - kicking_angle) / 180 * 100
            
            return preparation_score
        except:
            return 0.0
    
    def _calculate_running_efficiency(self, pose: Pose3D) -> float:
        """Analyser efficacité de course"""
        try:
            # Analyse posture de course
            spine_angle = pose.joint_angles.get('spine_lean', 0)
            balance = pose.joint_angles.get('balance_score', 50)
            
            # Angle optimal de course (légère inclinaison avant)
            optimal_lean = 10  # degrés
            lean_penalty = abs(spine_angle - optimal_lean) / optimal_lean
            
            # Score d'efficacité
            efficiency = (1 - lean_penalty) * (balance / 100) * 100
            
            return max(0, min(100, efficiency))
        except:
            return 50.0
    
    def _calculate_spine_angle(self, pose: Pose3D) -> float:
        """Calculer déviation de la colonne depuis la verticale"""
        
        try:
            left_shoulder = pose.get_keypoint(self.key_landmarks['left_shoulder'])[:2]
            right_shoulder = pose.get_keypoint(self.key_landmarks['right_shoulder'])[:2]
            left_hip = pose.get_keypoint(self.key_landmarks['left_hip'])[:2]
            right_hip = pose.get_keypoint(self.key_landmarks['right_hip'])[:2]
            
            # Points milieux épaules et hanches
            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                           (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
            
            # Vecteur colonne vertébrale
            spine_vector = np.array([shoulder_mid[0] - hip_mid[0],
                                    shoulder_mid[1] - hip_mid[1]])
            
            # Vecteur vertical de référence (vers le haut)
            vertical_vector = np.array([0, -1])
            
            # Calculer angle depuis la verticale
            cos_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector) + 1e-8
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            return math.degrees(angle)
        except:
            return 0.0
    
    def _calculate_balance_score(self, pose: Pose3D) -> float:
        """Calculer score d'équilibre basé sur symétrie et stabilité"""
        
        try:
            # Points clés
            left_ankle = pose.get_keypoint(self.key_landmarks['left_ankle'])
            right_ankle = pose.get_keypoint(self.key_landmarks['right_ankle'])
            left_hip = pose.get_keypoint(self.key_landmarks['left_hip'])
            right_hip = pose.get_keypoint(self.key_landmarks['right_hip'])
            
            # Centre de masse simplifié
            hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            ankle_center = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
            
            # Déviation latérale
            lateral_deviation = abs(hip_center[0] - ankle_center[0])
            
            # Espacement des pieds
            foot_spacing = abs(left_ankle[0] - right_ankle[0])
            
            # Score d'équilibre (0-100, plus élevé = meilleur)
            if foot_spacing > 0:
                balance_score = max(0, 100 - (lateral_deviation / foot_spacing) * 50)
            else:
                balance_score = 50  # Défaut si pieds superposés
            
            return balance_score
        except:
            return 50.0
    
    def draw_pose(self, image: np.ndarray, pose: Pose3D, 
                 show_angles: bool = False, show_com: bool = False) -> np.ndarray:
        """Dessiner pose avec keypoints, connexions et analyse avancée"""
        
        annotated_image = image.copy()
        
        # Dessiner landmarks
        for i, (x, y, z, visibility) in enumerate(pose.keypoints):
            if visibility > self.config.visibility_threshold:
                px, py = int(x), int(y)
                
                # Couleur selon type de point
                if i in [27, 28]:  # Chevilles
                    color = (255, 0, 0)  # Rouge
                elif i in [25, 26]:  # Genoux
                    color = (0, 255, 0)  # Vert
                elif i in [23, 24]:  # Hanches
                    color = (0, 0, 255)  # Bleu
                elif i in [11, 12]:  # Épaules
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (255, 255, 255)  # Blanc
                
                # Marquer points interpolés
                if i in pose.interpolated_points:
                    color = (128, 128, 128)  # Gris pour interpolés
                
                cv2.circle(annotated_image, (px, py), 4, color, -1)
                cv2.circle(annotated_image, (px, py), 6, (0, 0, 0), 1)
        
        # Dessiner connexions complètes
        connections = [
            # Torse
            (11, 12),  # Épaules
            (11, 23), (12, 24),  # Épaule vers hanche
            (23, 24),  # Hanches
            
            # Bras gauche
            (11, 13), (13, 15),  # Épaule -> coude -> poignet
            (15, 17), (15, 19), (15, 21),  # Poignet -> doigts
            
            # Bras droit
            (12, 14), (14, 16),  # Épaule -> coude -> poignet  
            (16, 18), (16, 20), (16, 22),  # Poignet -> doigts
            
            # Jambe gauche
            (23, 25), (25, 27),  # Hanche -> genou -> cheville
            (27, 29), (27, 31),  # Cheville -> talon/orteils
            
            # Jambe droite
            (24, 26), (26, 28),  # Hanche -> genou -> cheville
            (28, 30), (28, 32),  # Cheville -> talon/orteils
            
            # Visage
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),  # Yeux
            (0, 7), (0, 8),  # Oreilles
            (9, 10),  # Bouche
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(pose.keypoints) and end_idx < len(pose.keypoints):
                start_point = pose.keypoints[start_idx]
                end_point = pose.keypoints[end_idx]
                
                if (start_point[3] > self.config.visibility_threshold and 
                    end_point[3] > self.config.visibility_threshold):
                    start_px = (int(start_point[0]), int(start_point[1]))
                    end_px = (int(end_point[0]), int(end_point[1]))
                    cv2.line(annotated_image, start_px, end_px, (0, 255, 255), 2)
        
        # Dessiner centre de masse
        if show_com and pose.center_of_mass is not None:
            com_2d = pose.center_of_mass[:2]  # Projection 2D
            com_px = (int(com_2d[0]), int(com_2d[1]))
            cv2.circle(annotated_image, com_px, 8, (255, 0, 255), -1)
            cv2.putText(annotated_image, "COM", (com_px[0] + 10, com_px[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Afficher angles articulaires
        if show_angles and pose.joint_angles:
            y_offset = 30
            for angle_name, angle_value in pose.joint_angles.items():
                if isinstance(angle_value, (int, float)):
                    text = f"{angle_name}: {angle_value:.1f}°"
                    cv2.putText(annotated_image, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
        
        # Afficher métrique de confiance
        conf_text = f"Confiance: {pose.confidence:.2f}"
        cv2.putText(annotated_image, conf_text, (10, annotated_image.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Afficher ID du track
        track_text = f"Track: {pose.track_id}"
        cv2.putText(annotated_image, track_text, (10, annotated_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return annotated_image
    
    def analyze_running_form(self, poses: List[Pose3D]) -> Dict[str, Any]:
        """Analyser forme de course à partir d'une séquence de poses"""
        
        if len(poses) < 10:
            return {"error": "Poses insuffisantes pour analyse de course"}
        
        # Métriques au fil du temps
        knee_angles = []
        spine_angles = []
        balance_scores = []
        running_efficiency = []
        center_of_mass_trajectory = []
        
        for pose in poses:
            angles = pose.joint_angles
            knee_angles.append((angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2)
            spine_angles.append(angles.get('spine_lean', 0))
            balance_scores.append(angles.get('balance_score', 50))
            running_efficiency.append(angles.get('running_efficiency', 50))
            
            if pose.center_of_mass is not None:
                center_of_mass_trajectory.append(pose.center_of_mass)
        
        # Analyser stabilité COM
        com_stability = self._analyze_com_stability(center_of_mass_trajectory)
        
        # Détecter cycles de foulée
        stride_cycles = self._detect_stride_cycles(poses)
        
        return {
            "avg_knee_angle": float(np.mean(knee_angles)),
            "knee_angle_variability": float(np.std(knee_angles)),
            "avg_spine_lean": float(np.mean(spine_angles)),
            "spine_stability": float(100 - np.std(spine_angles)),
            "avg_balance_score": float(np.mean(balance_scores)),
            "balance_consistency": float(100 - np.std(balance_scores)),
            "running_efficiency": float(np.mean(running_efficiency)),
            "com_stability": com_stability,
            "stride_cycles_detected": len(stride_cycles),
            "avg_stride_length": self._calculate_avg_stride_length(stride_cycles),
            "cadence_estimate": self._estimate_cadence(stride_cycles, len(poses)),
            "overall_form_score": float(
                np.mean(balance_scores) * 0.3 +
                (100 - np.std(spine_angles)) * 0.2 +
                (100 - np.std(knee_angles)) * 0.2 +
                np.mean(running_efficiency) * 0.2 +
                com_stability * 0.1
            )
        }
    
    def _analyze_com_stability(self, com_trajectory: List[np.ndarray]) -> float:
        """Analyser stabilité du centre de masse"""
        if len(com_trajectory) < 5:
            return 50.0
        
        # Variance du mouvement COM
        com_array = np.array(com_trajectory)
        x_variance = np.var(com_array[:, 0])
        y_variance = np.var(com_array[:, 1])
        
        # Score de stabilité (inverse de la variance)
        stability = max(0, 100 - (x_variance + y_variance) * 10)
        return float(stability)
    
    def _detect_stride_cycles(self, poses: List[Pose3D]) -> List[Dict[str, int]]:
        """Détecter cycles de foulée basé sur mouvement des pieds"""
        cycles = []
        
        if len(poses) < 20:
            return cycles
        
        # Analyser position verticale des pieds
        left_ankle_y = [p.keypoints[self.key_landmarks['left_ankle'], 1] for p in poses]
        right_ankle_y = [p.keypoints[self.key_landmarks['right_ankle'], 1] for p in poses]
        
        # Détecter pics (appuis au sol)
        left_peaks = self._find_peaks(left_ankle_y)
        right_peaks = self._find_peaks(right_ankle_y)
        
        # Construire cycles alternés
        all_peaks = [(i, 'left') for i in left_peaks] + [(i, 'right') for i in right_peaks]
        all_peaks.sort()
        
        for i in range(len(all_peaks) - 1):
            cycle = {
                'start_frame': all_peaks[i][0],
                'end_frame': all_peaks[i + 1][0],
                'foot': all_peaks[i][1],
                'duration': all_peaks[i + 1][0] - all_peaks[i][0]
            }
            cycles.append(cycle)
        
        return cycles
    
    def _find_peaks(self, signal: List[float], min_distance: int = 10) -> List[int]:
        """Trouver pics locaux dans un signal"""
        peaks = []
        signal = np.array(signal)
        
        for i in range(min_distance, len(signal) - min_distance):
            if (signal[i] > signal[i-min_distance:i]).all() and \
               (signal[i] > signal[i+1:i+min_distance+1]).all():
                peaks.append(i)
        
        return peaks
    
    def _calculate_avg_stride_length(self, stride_cycles: List[Dict[str, int]]) -> float:
        """Calculer longueur moyenne de foulée"""
        if not stride_cycles:
            return 0.0
        
        durations = [cycle['duration'] for cycle in stride_cycles]
        return float(np.mean(durations))
    
    def _estimate_cadence(self, stride_cycles: List[Dict[str, int]], total_frames: int, fps: float = 30.0) -> float:
        """Estimer cadence (pas par minute)"""
        if not stride_cycles or total_frames == 0:
            return 0.0
        
        total_time_sec = total_frames / fps
        steps_per_second = len(stride_cycles) / total_time_sec
        steps_per_minute = steps_per_second * 60
        
        return float(steps_per_minute)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtenir statistiques de performance"""
        if not self.processing_times:
            return {}
        
        success_rate = np.mean(self.success_rate) * 100 if self.success_rate else 0
        
        return {
            'avg_processing_time_ms': float(np.mean(self.processing_times)),
            'max_processing_time_ms': float(np.max(self.processing_times)),
            'min_processing_time_ms': float(np.min(self.processing_times)),
            'success_rate_percent': float(success_rate),
            'total_extractions': len(self.processing_times),
            'mediapipe_available': MEDIAPIPE_AVAILABLE,
            'scipy_available': SCIPY_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'gpu_enabled': self.config.use_gpu and TORCH_AVAILABLE,
            'batch_processing': self.batch_processing,
            'temporal_smoothing': self.config.enable_temporal_smoothing
        }
    
    def export_poses_to_json(self, poses_by_track: Dict[int, List[Pose3D]], 
                            output_path: str) -> bool:
        """Exporter poses vers JSON"""
        try:
            export_data = {
                'metadata': {
                    'extractor_config': {
                        'model_complexity': self.config.model_complexity,
                        'enable_segmentation': self.config.enable_segmentation,
                        'smooth_landmarks': self.config.smooth_landmarks,
                        'temporal_smoothing': self.config.enable_temporal_smoothing
                    },
                    'performance_stats': self.get_performance_stats(),
                    'total_tracks': len(poses_by_track),
                    'total_poses': sum(len(poses) for poses in poses_by_track.values())
                },
                'poses_by_track': {}
            }
            
            for track_id, poses in poses_by_track.items():
                export_data['poses_by_track'][track_id] = []
                
                for pose in poses:
                    pose_data = {
                        'frame_number': pose.frame_number,
                        'confidence': pose.confidence,
                        'keypoints': pose.keypoints.tolist(),
                        'world_landmarks': pose.world_landmarks.tolist() if pose.world_landmarks is not None else None,
                        'joint_angles': pose.joint_angles,
                        'center_of_mass': pose.center_of_mass.tolist() if pose.center_of_mass is not None else None,
                        'body_orientation': pose.body_orientation.tolist() if pose.body_orientation is not None else None,
                        'interpolated_points': pose.interpolated_points
                    }
                    export_data['poses_by_track'][track_id].append(pose_data)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Poses exportées vers {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur export JSON: {e}")
            return False


# Fonctions utilitaires
def create_pose_extractor(config: Optional[PoseExtractionConfig] = None) -> PoseExtractor:
    """Créer instance d'extracteur de pose"""
    return PoseExtractor(config)


if __name__ == "__main__":
    # Test de l'extracteur
    config = PoseExtractionConfig(
        model_complexity=2,
        enable_segmentation=True,
        batch_size=4,
        enable_temporal_smoothing=True
    )
    
    extractor = PoseExtractor(config)
    
    # Image de test
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test extraction
    pose = extractor.extract_pose(test_image, track_id=1, frame_number=0)
    
    if pose:
        print(f"Pose extraite avec confiance: {pose.confidence:.2f}")
        print(f"Angles articulaires: {len(pose.joint_angles)} calculés")
        print(f"Points interpolés: {len(pose.interpolated_points)}")
    else:
        print("Échec extraction pose")
    
    # Statistiques de performance
    stats = extractor.get_performance_stats()
    print(f"Statistiques: {stats}")