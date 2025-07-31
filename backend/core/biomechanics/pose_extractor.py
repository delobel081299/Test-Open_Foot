import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math

from backend.core.tracking.byte_tracker import Track
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Pose3D:
    keypoints: np.ndarray  # 33 keypoints x 4 (x, y, z, visibility)
    confidence: float
    track_id: int
    frame_number: int
    
    def get_keypoint(self, landmark_id: int) -> Tuple[float, float, float, float]:
        """Get specific keypoint (x, y, z, visibility)"""
        return tuple(self.keypoints[landmark_id])
    
    def is_visible(self, landmark_id: int, threshold: float = 0.5) -> bool:
        """Check if keypoint is visible"""
        return self.keypoints[landmark_id][3] > threshold

class PoseExtractor:
    """MediaPipe-based pose extraction for biomechanical analysis"""
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose_detector = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks
        )
        
        # Key landmark indices for football analysis
        self.key_landmarks = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
    
    def extract_pose(self, image: np.ndarray, track_id: int, frame_number: int) -> Optional[Pose3D]:
        """Extract pose from single image crop"""
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose_detector.process(rgb_image)
        
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = np.zeros((33, 4))  # x, y, z, visibility
            
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]
            
            # Calculate overall confidence
            confidence = np.mean(keypoints[:, 3])
            
            return Pose3D(
                keypoints=keypoints,
                confidence=confidence,
                track_id=track_id,
                frame_number=frame_number
            )
        
        return None
    
    def extract_poses_from_tracks(
        self,
        tracks: List[Track],
        frames: List[np.ndarray]
    ) -> Dict[int, List[Pose3D]]:
        """Extract poses for all tracked players"""
        
        logger.info(f"Extracting poses for {len(tracks)} tracks")
        
        poses_by_track = {}
        
        for track in tracks:
            track_poses = []
            
            for detection, frame_num in zip(track.detections, track.frames):
                if frame_num < len(frames):
                    frame = frames[frame_num]
                    
                    # Crop player region
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        player_crop = frame[y1:y2, x1:x2]
                        
                        pose = self.extract_pose(player_crop, track.track_id, frame_num)
                        if pose:
                            # Convert relative coordinates to absolute
                            pose.keypoints[:, 0] = pose.keypoints[:, 0] * (x2 - x1) + x1
                            pose.keypoints[:, 1] = pose.keypoints[:, 1] * (y2 - y1) + y1
                            track_poses.append(pose)
            
            if track_poses:
                poses_by_track[track.track_id] = track_poses
        
        logger.info(f"Extracted poses for {len(poses_by_track)} players")
        return poses_by_track
    
    def calculate_joint_angles(self, pose: Pose3D) -> Dict[str, float]:
        """Calculate key joint angles for biomechanical analysis"""
        
        angles = {}
        
        try:
            # Knee angles
            angles['left_knee'] = self._calculate_angle(
                pose.get_keypoint(self.key_landmarks['left_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['left_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['left_ankle'])[:2]
            )
            
            angles['right_knee'] = self._calculate_angle(
                pose.get_keypoint(self.key_landmarks['right_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['right_knee'])[:2],
                pose.get_keypoint(self.key_landmarks['right_ankle'])[:2]
            )
            
            # Hip angles
            angles['left_hip'] = self._calculate_angle(
                pose.get_keypoint(self.key_landmarks['left_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['left_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['left_knee'])[:2]
            )
            
            angles['right_hip'] = self._calculate_angle(
                pose.get_keypoint(self.key_landmarks['right_shoulder'])[:2],
                pose.get_keypoint(self.key_landmarks['right_hip'])[:2],
                pose.get_keypoint(self.key_landmarks['right_knee'])[:2]
            )
            
            # Spine angle (deviation from vertical)
            spine_angle = self._calculate_spine_angle(pose)
            angles['spine_lean'] = spine_angle
            
            # Balance analysis
            balance_score = self._calculate_balance_score(pose)
            angles['balance_score'] = balance_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate some joint angles: {str(e)}")
        
        return angles
    
    def _calculate_angle(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        point3: Tuple[float, float]
    ) -> float:
        """Calculate angle between three points"""
        
        # Vector from point2 to point1
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        # Vector from point2 to point3
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return math.degrees(angle)
    
    def _calculate_spine_angle(self, pose: Pose3D) -> float:
        """Calculate spine deviation from vertical"""
        
        left_shoulder = pose.get_keypoint(self.key_landmarks['left_shoulder'])[:2]
        right_shoulder = pose.get_keypoint(self.key_landmarks['right_shoulder'])[:2]
        left_hip = pose.get_keypoint(self.key_landmarks['left_hip'])[:2]
        right_hip = pose.get_keypoint(self.key_landmarks['right_hip'])[:2]
        
        # Calculate shoulder and hip midpoints
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                   (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate spine vector
        spine_vector = np.array([shoulder_mid[0] - hip_mid[0],
                                shoulder_mid[1] - hip_mid[1]])
        
        # Vertical reference vector (pointing up)
        vertical_vector = np.array([0, -1])
        
        # Calculate angle from vertical
        cos_angle = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector) + 1e-8
        )
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return math.degrees(angle)
    
    def _calculate_balance_score(self, pose: Pose3D) -> float:
        """Calculate balance score based on pose symmetry and stability"""
        
        # Get key points
        left_ankle = pose.get_keypoint(self.key_landmarks['left_ankle'])
        right_ankle = pose.get_keypoint(self.key_landmarks['right_ankle'])
        left_hip = pose.get_keypoint(self.key_landmarks['left_hip'])
        right_hip = pose.get_keypoint(self.key_landmarks['right_hip'])
        
        # Calculate center of mass (simplified)
        hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        ankle_center = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
        
        # Calculate lateral deviation
        lateral_deviation = abs(hip_center[0] - ankle_center[0])
        
        # Calculate foot spacing
        foot_spacing = abs(left_ankle[0] - right_ankle[0])
        
        # Balance score (0-100, higher is better)
        balance_score = max(0, 100 - (lateral_deviation / max(foot_spacing, 1)) * 100)
        
        return balance_score
    
    def draw_pose(self, image: np.ndarray, pose: Pose3D) -> np.ndarray:
        """Draw pose keypoints and connections on image"""
        
        annotated_image = image.copy()
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        
        # Draw landmarks
        for i, (x, y, z, visibility) in enumerate(pose.keypoints):
            if visibility > 0.5:
                px, py = int(x), int(y)
                cv2.circle(annotated_image, (px, py), 3, (0, 255, 0), -1)
        
        # Draw connections (simplified)
        connections = [
            (11, 12),  # shoulders
            (11, 23), (12, 24),  # shoulder to hip
            (23, 24),  # hips
            (25, 27), (26, 28),  # knee to ankle
            (23, 25), (24, 26),  # hip to knee
        ]
        
        for start_idx, end_idx in connections:
            start_point = pose.keypoints[start_idx]
            end_point = pose.keypoints[end_idx]
            
            if start_point[3] > 0.5 and end_point[3] > 0.5:
                start_px = (int(start_point[0]), int(start_point[1]))
                end_px = (int(end_point[0]), int(end_point[1]))
                cv2.line(annotated_image, start_px, end_px, (255, 0, 0), 2)
        
        return annotated_image
    
    def analyze_running_form(self, poses: List[Pose3D]) -> Dict[str, Any]:
        """Analyze running form from pose sequence"""
        
        if len(poses) < 10:
            return {"error": "Insufficient poses for running analysis"}
        
        # Calculate metrics over time
        knee_angles = []
        spine_angles = []
        balance_scores = []
        
        for pose in poses:
            angles = self.calculate_joint_angles(pose)
            knee_angles.append((angles.get('left_knee', 0) + angles.get('right_knee', 0)) / 2)
            spine_angles.append(angles.get('spine_lean', 0))
            balance_scores.append(angles.get('balance_score', 0))
        
        return {
            "avg_knee_angle": np.mean(knee_angles),
            "knee_angle_variability": np.std(knee_angles),
            "avg_spine_lean": np.mean(spine_angles),
            "spine_stability": 100 - np.std(spine_angles),
            "avg_balance_score": np.mean(balance_scores),
            "balance_consistency": 100 - np.std(balance_scores),
            "overall_form_score": (
                np.mean(balance_scores) * 0.4 +
                (100 - np.std(spine_angles)) * 0.3 +
                (100 - np.std(knee_angles)) * 0.3
            )
        }