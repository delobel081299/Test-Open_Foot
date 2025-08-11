import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN

from backend.utils.logger import setup_logger
from .advanced_detector import Detection
from .yolo_detector import YOLODetector

logger = setup_logger(__name__)

class BallState(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"
    OCCLUDED = "occluded"
    OUT_OF_BOUNDS = "out_of_bounds"
    UNCERTAIN = "uncertain"

class ContactType(Enum):
    FOOT = "foot"
    HEAD = "head"
    BODY = "body"
    NONE = "none"

@dataclass
class BallDetection:
    """Enhanced ball detection with comprehensive information"""
    position: Tuple[float, float]  # x, y pixel coordinates
    velocity: Tuple[float, float]  # vx, vy pixels per frame
    acceleration: Tuple[float, float]  # ax, ay pixels per frame²
    confidence: float
    state: BallState
    
    # Physical properties
    radius: float
    height_estimate: Optional[float] = None  # Estimated height above ground
    
    # Possession and contact
    possession_player_id: Optional[int] = None
    possession_confidence: float = 0.0
    last_contact_type: ContactType = ContactType.NONE
    last_contact_frame: Optional[int] = None
    
    # Trajectory prediction
    predicted_trajectory: Optional[List[Tuple[float, float]]] = None
    trajectory_confidence: float = 0.0
    
    # Validation scores
    yolo_confidence: float = 0.0
    circle_validation_score: float = 0.0
    template_match_score: float = 0.0
    temporal_consistency: float = 0.0

@dataclass
class BallTemplate:
    """Ball template for template matching"""
    template: np.ndarray
    scale: float
    confidence_threshold: float

class KalmanBallTracker:
    """Kalman filter for smooth ball tracking"""
    
    def __init__(self):
        # State vector: [x, y, vx, vy, ax, ay]
        self.kf = cv2.KalmanFilter(6, 2)
        
        # Transition matrix (constant acceleration model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        # Error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1000
        
        self.initialized = False
    
    def initialize(self, position: Tuple[float, float]):
        """Initialize Kalman filter with first detection"""
        x, y = position
        self.kf.statePre = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """Predict next ball position"""
        if not self.initialized:
            return (0, 0)
        
        prediction = self.kf.predict()
        return (float(prediction[0]), float(prediction[1]))
    
    def update(self, measurement: Tuple[float, float]) -> np.ndarray:
        """Update filter with new measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return self.kf.statePost
        
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kf.correct(measurement_array)
        return self.kf.statePost
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        if not self.initialized:
            return (0, 0)
        
        state = self.kf.statePost
        return (float(state[2]), float(state[3]))
    
    def get_acceleration(self) -> Tuple[float, float]:
        """Get current acceleration estimate"""
        if not self.initialized:
            return (0, 0)
        
        state = self.kf.statePost
        return (float(state[4]), float(state[5]))

class BallDetector:
    """High-precision ball detector combining multiple techniques"""
    
    def __init__(self, 
                 yolo_model_path: str = "models/yolov10/yolov10x.pt",
                 config: Optional[Dict] = None):
        
        self.config = config or self._default_config()
        
        # Initialize YOLO detector
        self.yolo_detector = YOLODetector(yolo_model_path)
        
        # Kalman tracker
        self.kalman_tracker = KalmanBallTracker()
        
        # Frame history for temporal analysis
        self.frame_history = deque(maxlen=self.config['temporal']['history_length'])
        self.detection_history = deque(maxlen=self.config['temporal']['history_length'])
        
        # Optical flow tracker
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.optical_flow_points = None
        
        # Ball templates for template matching
        self.ball_templates = self._initialize_ball_templates()
        
        # State tracking
        self.current_ball_state = BallState.UNCERTAIN
        self.frames_since_detection = 0
        self.frames_since_contact = 0
        
        # Performance tracking
        self.detection_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'yolo_detections': 0,
            'circle_validations': 0,
            'template_matches': 0,
            'predicted_positions': 0
        }
        
        logger.info("Ball detector initialized with multi-technique approach")
    
    def _default_config(self) -> Dict:
        """Default configuration for ball detection"""
        return {
            'yolo': {
                'confidence_threshold': 0.3,
                'ball_class_id': 32,  # sports ball in COCO
                'min_area': 10,
                'max_area': 1000
            },
            'hough_circles': {
                'dp': 1,
                'min_dist': 30,
                'param1': 50,
                'param2': 30,
                'min_radius': 3,
                'max_radius': 50
            },
            'template_matching': {
                'method': cv2.TM_CCOEFF_NORMED,
                'threshold': 0.6,
                'scales': [0.5, 0.7, 1.0, 1.3, 1.5]
            },
            'optical_flow': {
                'max_distance': 100,
                'quality_threshold': 0.1
            },
            'possession': {
                'max_distance': 80,
                'contact_threshold': 15,
                'min_frames_for_possession': 3
            },
            'trajectory': {
                'min_points': 5,
                'prediction_frames': 10,
                'gravity': 9.81  # pixels per frame² (needs calibration)
            },
            'temporal': {
                'history_length': 30,
                'consistency_threshold': 0.7,
                'max_frames_without_detection': 10
            },
            'validation': {
                'min_circularity': 0.6,
                'color_range_hsv': [(10, 50, 50), (25, 255, 255)],  # Orange/yellow ball
                'size_consistency_threshold': 0.3
            }
        }
    
    def _initialize_ball_templates(self) -> List[BallTemplate]:
        """Initialize ball templates for template matching"""
        templates = []
        
        # Create synthetic ball templates of different sizes
        for radius in [8, 12, 16, 20, 25]:
            # Create a simple circular template
            template_size = radius * 2 + 4
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            
            # Draw filled circle
            cv2.circle(template, (template_size//2, template_size//2), radius, 255, -1)
            
            # Add some gradient for more realistic appearance
            for i in range(template_size):
                for j in range(template_size):
                    dist = np.sqrt((i - template_size//2)**2 + (j - template_size//2)**2)
                    if dist <= radius:
                        intensity = max(0, 255 - int(dist * 20))
                        template[i, j] = min(255, template[i, j] + intensity)
            
            ball_template = BallTemplate(
                template=template,
                scale=radius / 16.0,  # Normalized scale
                confidence_threshold=0.6
            )
            templates.append(ball_template)
        
        logger.info(f"Initialized {len(templates)} ball templates")
        return templates
    
    def detect_ball(self, 
                   frame: np.ndarray,
                   players: Optional[List[Detection]] = None,
                   frame_idx: int = 0) -> Optional[BallDetection]:
        """Main ball detection method combining all techniques"""
        
        self.detection_stats['total_frames'] += 1
        
        # Store frame in history
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_history.append(gray_frame)
        
        # Step 1: YOLO initial detection
        yolo_candidates = self._yolo_ball_detection(frame)
        
        # Step 2: Hough Circle validation
        circle_candidates = self._hough_circle_detection(gray_frame)
        
        # Step 3: Template matching for difficult cases
        template_candidates = self._template_matching_detection(gray_frame)
        
        # Step 4: Optical flow prediction
        flow_prediction = self._optical_flow_prediction(gray_frame)
        
        # Combine all candidates
        all_candidates = self._combine_candidates(
            yolo_candidates, circle_candidates, template_candidates, flow_prediction
        )
        
        # Select best candidate
        best_detection = self._select_best_candidate(all_candidates, frame, players)
        
        # Step 5: Kalman filter smoothing
        if best_detection:
            smoothed_detection = self._apply_kalman_smoothing(best_detection)
            
            # Add advanced features
            enhanced_detection = self._add_advanced_features(
                smoothed_detection, frame, players, frame_idx
            )
            
            self.detection_history.append(enhanced_detection)
            self.frames_since_detection = 0
            self.current_ball_state = BallState.DETECTED
            self.detection_stats['successful_detections'] += 1
            
            return enhanced_detection
        
        else:
            # Handle cases where no ball is detected
            self.frames_since_detection += 1
            
            if self.frames_since_detection <= self.config['temporal']['max_frames_without_detection']:
                # Try to predict position based on trajectory
                predicted_detection = self._predict_ball_position(frame, players, frame_idx)
                if predicted_detection:
                    self.detection_history.append(predicted_detection)
                    self.current_ball_state = BallState.PREDICTED
                    self.detection_stats['predicted_positions'] += 1
                    return predicted_detection
            
            self.current_ball_state = BallState.UNCERTAIN
            return None
    
    def _yolo_ball_detection(self, frame: np.ndarray) -> List[Dict]:
        """YOLO-based ball detection"""
        candidates = []
        
        try:
            detections = self.yolo_detector.detect(frame)
            
            for detection in detections:
                if detection.class_name == 'sports ball':
                    x1, y1, x2, y2 = detection.bbox
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Filter by area
                    if (self.config['yolo']['min_area'] <= area <= 
                        self.config['yolo']['max_area']):
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        radius = max(x2 - x1, y2 - y1) / 2
                        
                        candidate = {
                            'position': (center_x, center_y),
                            'radius': radius,
                            'confidence': detection.confidence,
                            'source': 'yolo',
                            'bbox': detection.bbox
                        }
                        candidates.append(candidate)
                        self.detection_stats['yolo_detections'] += 1
            
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        
        return candidates
    
    def _hough_circle_detection(self, gray_frame: np.ndarray) -> List[Dict]:
        """Hough Circle Transform for ball validation"""
        candidates = []
        
        try:
            # Preprocess for circle detection
            blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            
            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.config['hough_circles']['dp'],
                minDist=self.config['hough_circles']['min_dist'],
                param1=self.config['hough_circles']['param1'],
                param2=self.config['hough_circles']['param2'],
                minRadius=self.config['hough_circles']['min_radius'],
                maxRadius=self.config['hough_circles']['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Validate circle quality
                    validation_score = self._validate_circle_quality(gray_frame, x, y, r)
                    
                    if validation_score > 0.5:
                        candidate = {
                            'position': (float(x), float(y)),
                            'radius': float(r),
                            'confidence': validation_score,
                            'source': 'hough',
                            'validation_score': validation_score
                        }
                        candidates.append(candidate)
                        self.detection_stats['circle_validations'] += 1
            
        except Exception as e:
            logger.warning(f"Hough circle detection failed: {e}")
        
        return candidates
    
    def _validate_circle_quality(self, gray_frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Validate circle quality for ball detection"""
        h, w = gray_frame.shape
        
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0
        
        # Extract circle region
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        circle_region = cv2.bitwise_and(gray_frame, mask)
        
        # Check circularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Check color consistency (for orange/yellow ball)
        hsv_region = cv2.cvtColor(
            cv2.cvtColor(circle_region, cv2.COLOR_GRAY2BGR), 
            cv2.COLOR_BGR2HSV
        )
        
        # Color validation score
        color_score = self._validate_ball_color(hsv_region, mask)
        
        # Combine scores
        quality_score = 0.6 * circularity + 0.4 * color_score
        return min(1.0, max(0.0, quality_score))
    
    def _validate_ball_color(self, hsv_image: np.ndarray, mask: np.ndarray) -> float:
        """Validate ball color (orange/yellow range)"""
        try:
            lower_bound, upper_bound = self.config['validation']['color_range_hsv']
            color_mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
            
            # Calculate overlap with circle mask
            overlap = cv2.bitwise_and(color_mask, mask)
            overlap_ratio = np.sum(overlap > 0) / max(1, np.sum(mask > 0))
            
            return overlap_ratio
            
        except Exception:
            return 0.5  # Neutral score if color validation fails
    
    def _template_matching_detection(self, gray_frame: np.ndarray) -> List[Dict]:
        """Template matching for difficult ball detection cases"""
        candidates = []
        
        try:
            for template_obj in self.ball_templates:
                template = template_obj.template
                
                # Multi-scale template matching
                for scale in self.config['template_matching']['scales']:
                    scaled_template = cv2.resize(
                        template, 
                        None, 
                        fx=scale, 
                        fy=scale, 
                        interpolation=cv2.INTER_CUBIC
                    )
                    
                    if (scaled_template.shape[0] >= gray_frame.shape[0] or 
                        scaled_template.shape[1] >= gray_frame.shape[1]):
                        continue
                    
                    # Template matching
                    result = cv2.matchTemplate(
                        gray_frame, 
                        scaled_template, 
                        self.config['template_matching']['method']
                    )
                    
                    # Find matches above threshold
                    locations = np.where(result >= self.config['template_matching']['threshold'])
                    
                    for pt in zip(*locations[::-1]):
                        match_score = result[pt[1], pt[0]]
                        
                        center_x = pt[0] + scaled_template.shape[1] // 2
                        center_y = pt[1] + scaled_template.shape[0] // 2
                        radius = min(scaled_template.shape) // 2
                        
                        candidate = {
                            'position': (float(center_x), float(center_y)),
                            'radius': float(radius),
                            'confidence': float(match_score),
                            'source': 'template',
                            'template_scale': scale,
                            'match_score': float(match_score)
                        }
                        candidates.append(candidate)
                        self.detection_stats['template_matches'] += 1
            
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
        
        return candidates
    
    def _optical_flow_prediction(self, gray_frame: np.ndarray) -> Optional[Dict]:
        """Optical flow-based prediction"""
        if len(self.frame_history) < 2 or not self.detection_history:
            return None
        
        try:
            prev_frame = self.frame_history[-2]
            last_detection = self.detection_history[-1]
            
            # Initialize tracking points if needed
            if self.optical_flow_points is None:
                x, y = last_detection.position
                self.optical_flow_points = np.array([[x, y]], dtype=np.float32)
            
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_frame, 
                gray_frame, 
                self.optical_flow_points, 
                None, 
                **self.lk_params
            )
            
            # Validate tracking quality
            if status[0] == 1 and error[0] < self.config['optical_flow']['quality_threshold']:
                predicted_x, predicted_y = new_points[0]
                
                # Update tracking points
                self.optical_flow_points = new_points
                
                # Calculate movement distance
                last_x, last_y = last_detection.position
                distance = np.sqrt((predicted_x - last_x)**2 + (predicted_y - last_y)**2)
                
                # Validate reasonable movement
                if distance <= self.config['optical_flow']['max_distance']:
                    return {
                        'position': (float(predicted_x), float(predicted_y)),
                        'radius': last_detection.radius,
                        'confidence': 0.7 * (1.0 - error[0]),
                        'source': 'optical_flow',
                        'movement_distance': float(distance)
                    }
            
            # Reset tracking if failed
            self.optical_flow_points = None
            
        except Exception as e:
            logger.warning(f"Optical flow prediction failed: {e}")
        
        return None
    
    def _combine_candidates(self, *candidate_lists) -> List[Dict]:
        """Combine candidates from all detection methods"""
        all_candidates = []
        
        for candidates in candidate_lists:
            if candidates is None:
                continue
            
            if isinstance(candidates, dict):
                all_candidates.append(candidates)
            elif isinstance(candidates, list):
                all_candidates.extend(candidates)
        
        # Remove duplicates using spatial clustering
        if len(all_candidates) > 1:
            all_candidates = self._remove_duplicate_candidates(all_candidates)
        
        return all_candidates
    
    def _remove_duplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate candidates using spatial clustering"""
        if len(candidates) <= 1:
            return candidates
        
        # Extract positions
        positions = np.array([c['position'] for c in candidates])
        
        # Use DBSCAN to cluster nearby detections
        clustering = DBSCAN(eps=20, min_samples=1).fit(positions)
        labels = clustering.labels_
        
        unique_candidates = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            
            # Get all candidates in this cluster
            cluster_candidates = [c for i, c in enumerate(candidates) if labels[i] == label]
            
            # Select best candidate in cluster (highest confidence)
            best_candidate = max(cluster_candidates, key=lambda c: c['confidence'])
            unique_candidates.append(best_candidate)
        
        return unique_candidates
    
    def _select_best_candidate(self, 
                              candidates: List[Dict], 
                              frame: np.ndarray,
                              players: Optional[List[Detection]] = None) -> Optional[Dict]:
        """Select the best ball candidate"""
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            score = self._score_candidate(candidate, frame, players)
            candidate['total_score'] = score
            scored_candidates.append(candidate)
        
        # Return best scoring candidate
        best_candidate = max(scored_candidates, key=lambda c: c['total_score'])
        
        # Apply minimum threshold
        if best_candidate['total_score'] > 0.3:
            return best_candidate
        
        return None
    
    def _score_candidate(self, 
                        candidate: Dict, 
                        frame: np.ndarray,
                        players: Optional[List[Detection]] = None) -> float:
        """Score a ball candidate"""
        score = 0.0
        
        # Base confidence from detection method
        base_confidence = candidate['confidence']
        score += 0.4 * base_confidence
        
        # Bonus for multi-method detection
        if candidate['source'] == 'yolo':
            score += 0.3
        elif candidate['source'] == 'hough':
            score += 0.2
        elif candidate['source'] == 'template':
            score += 0.15
        elif candidate['source'] == 'optical_flow':
            score += 0.1
        
        # Temporal consistency bonus
        if self.detection_history:
            temporal_score = self._calculate_temporal_consistency(candidate)
            score += 0.2 * temporal_score
        
        # Size consistency
        size_score = self._validate_ball_size(candidate, frame.shape)
        score += 0.1 * size_score
        
        # Penalty for being too close to players (possible false positive)
        if players:
            player_penalty = self._calculate_player_interference_penalty(candidate, players)
            score -= 0.1 * player_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_temporal_consistency(self, candidate: Dict) -> float:
        """Calculate temporal consistency with previous detections"""
        if not self.detection_history:
            return 0.5
        
        recent_detections = list(self.detection_history)[-5:]  # Last 5 detections
        
        # Calculate average distance to recent positions
        candidate_pos = np.array(candidate['position'])
        distances = []
        
        for detection in recent_detections:
            detection_pos = np.array(detection.position)
            distance = np.linalg.norm(candidate_pos - detection_pos)
            distances.append(distance)
        
        if not distances:
            return 0.5
        
        avg_distance = np.mean(distances)
        
        # Convert distance to consistency score (closer = more consistent)
        consistency = max(0.0, 1.0 - avg_distance / 100.0)
        
        return consistency
    
    def _validate_ball_size(self, candidate: Dict, frame_shape: Tuple[int, int]) -> float:
        """Validate ball size reasonableness"""
        radius = candidate['radius']
        
        # Expected ball size range (in pixels)
        min_radius = 3
        max_radius = min(frame_shape) // 20  # Not more than 1/20 of frame dimension
        
        if min_radius <= radius <= max_radius:
            # Additional check: size should be consistent with previous detections
            if self.detection_history:
                recent_sizes = [d.radius for d in list(self.detection_history)[-5:]]
                avg_size = np.mean(recent_sizes)
                size_difference = abs(radius - avg_size) / avg_size
                
                consistency = max(0.0, 1.0 - size_difference / 0.5)  # 50% tolerance
                return consistency
            else:
                return 1.0
        
        return 0.0
    
    def _calculate_player_interference_penalty(self, 
                                             candidate: Dict, 
                                             players: List[Detection]) -> float:
        """Calculate penalty for ball being too close to player centers"""
        candidate_pos = np.array(candidate['position'])
        
        min_distance = float('inf')
        for player in players:
            x1, y1, x2, y2 = player.bbox
            player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            distance = np.linalg.norm(candidate_pos - player_center)
            min_distance = min(min_distance, distance)
        
        # If very close to player center, it might be a false positive (head/body part)
        if min_distance < 15:
            return 1.0  # High penalty
        elif min_distance < 30:
            return 0.5  # Medium penalty
        else:
            return 0.0  # No penalty
    
    def _apply_kalman_smoothing(self, candidate: Dict) -> BallDetection:
        """Apply Kalman filter smoothing"""
        position = candidate['position']
        
        # Update Kalman filter
        self.kalman_tracker.update(position)
        
        # Get smoothed estimates
        smoothed_velocity = self.kalman_tracker.get_velocity()
        smoothed_acceleration = self.kalman_tracker.get_acceleration()
        
        # Create ball detection object
        detection = BallDetection(
            position=position,
            velocity=smoothed_velocity,
            acceleration=smoothed_acceleration,
            confidence=candidate['confidence'],
            state=BallState.DETECTED,
            radius=candidate['radius'],
            yolo_confidence=candidate.get('confidence', 0.0) if candidate['source'] == 'yolo' else 0.0,
            circle_validation_score=candidate.get('validation_score', 0.0) if candidate['source'] == 'hough' else 0.0,
            template_match_score=candidate.get('match_score', 0.0) if candidate['source'] == 'template' else 0.0,
            temporal_consistency=self._calculate_temporal_consistency(candidate)
        )
        
        return detection
    
    def _add_advanced_features(self, 
                              detection: BallDetection,
                              frame: np.ndarray,
                              players: Optional[List[Detection]] = None,
                              frame_idx: int = 0) -> BallDetection:
        """Add advanced features like possession, trajectory prediction"""
        
        # Detect possession
        if players:
            possession_info = self._detect_possession(detection, players)
            detection.possession_player_id = possession_info.get('player_id')
            detection.possession_confidence = possession_info.get('confidence', 0.0)
        
        # Detect contact
        contact_info = self._detect_ball_contact(detection, players, frame_idx)
        detection.last_contact_type = contact_info.get('type', ContactType.NONE)
        detection.last_contact_frame = contact_info.get('frame_idx')
        
        # Predict trajectory
        if len(self.detection_history) >= 3:
            trajectory_info = self._predict_trajectory(detection)
            detection.predicted_trajectory = trajectory_info.get('trajectory')
            detection.trajectory_confidence = trajectory_info.get('confidence', 0.0)
        
        # Estimate height
        detection.height_estimate = self._estimate_ball_height(detection, frame)
        
        return detection
    
    def _detect_possession(self, 
                          ball_detection: BallDetection, 
                          players: List[Detection]) -> Dict:
        """Detect which player has possession of the ball"""
        if not players:
            return {}
        
        ball_pos = np.array(ball_detection.position)
        min_distance = float('inf')
        closest_player_id = None
        
        for i, player in enumerate(players):
            x1, y1, x2, y2 = player.bbox
            
            # Calculate distance to player's feet (bottom of bbox)
            foot_pos = np.array([(x1 + x2) / 2, y2])
            distance = np.linalg.norm(ball_pos - foot_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_player_id = player.track_id or i
        
        # Determine possession based on distance
        max_possession_distance = self.config['possession']['max_distance']
        
        if min_distance <= max_possession_distance:
            confidence = max(0.0, 1.0 - min_distance / max_possession_distance)
            
            return {
                'player_id': closest_player_id,
                'confidence': confidence,
                'distance': min_distance
            }
        
        return {'player_id': None, 'confidence': 0.0}
    
    def _detect_ball_contact(self, 
                            ball_detection: BallDetection,
                            players: Optional[List[Detection]] = None,
                            frame_idx: int = 0) -> Dict:
        """Detect ball contact with players"""
        if not players or len(self.detection_history) < 2:
            return {'type': ContactType.NONE}
        
        # Get ball velocity magnitude
        velocity = np.linalg.norm(ball_detection.velocity)
        
        # Check for sudden velocity changes (indicating contact)
        prev_detection = self.detection_history[-1]
        prev_velocity = np.linalg.norm(prev_detection.velocity)
        
        velocity_change = abs(velocity - prev_velocity)
        
        # If significant velocity change and ball is close to a player
        contact_threshold = self.config['possession']['contact_threshold']
        
        if velocity_change > 20:  # Significant velocity change
            ball_pos = np.array(ball_detection.position)
            
            for player in players:
                x1, y1, x2, y2 = player.bbox
                
                # Check different body parts
                head_pos = np.array([(x1 + x2) / 2, y1 + (y2 - y1) * 0.15])
                foot_pos = np.array([(x1 + x2) / 2, y2])
                body_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                
                head_distance = np.linalg.norm(ball_pos - head_pos)
                foot_distance = np.linalg.norm(ball_pos - foot_pos)
                body_distance = np.linalg.norm(ball_pos - body_center)
                
                if foot_distance <= contact_threshold:
                    return {'type': ContactType.FOOT, 'frame_idx': frame_idx}
                elif head_distance <= contact_threshold:
                    return {'type': ContactType.HEAD, 'frame_idx': frame_idx}
                elif body_distance <= contact_threshold:
                    return {'type': ContactType.BODY, 'frame_idx': frame_idx}
        
        return {'type': ContactType.NONE}
    
    def _predict_trajectory(self, ball_detection: BallDetection) -> Dict:
        """Predict ball trajectory using physics"""
        if len(self.detection_history) < self.config['trajectory']['min_points']:
            return {}
        
        try:
            # Get recent positions and timestamps
            recent_detections = list(self.detection_history)[-10:]
            positions = np.array([d.position for d in recent_detections])
            
            # Fit parabolic trajectory (accounting for gravity)
            trajectory, confidence = self._fit_parabolic_trajectory(positions)
            
            if confidence > 0.5:
                # Predict future positions
                future_trajectory = self._extrapolate_trajectory(
                    trajectory, 
                    self.config['trajectory']['prediction_frames']
                )
                
                return {
                    'trajectory': future_trajectory,
                    'confidence': confidence,
                    'model_params': trajectory
                }
        
        except Exception as e:
            logger.warning(f"Trajectory prediction failed: {e}")
        
        return {}
    
    def _fit_parabolic_trajectory(self, positions: np.ndarray) -> Tuple[Dict, float]:
        """Fit parabolic trajectory to ball positions"""
        if len(positions) < 3:
            return {}, 0.0
        
        try:
            # Extract x and y coordinates
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            # Fit quadratic polynomial: y = ax² + bx + c
            # Using time as parameter (frame indices)
            t = np.arange(len(positions))
            
            # Fit x(t) and y(t) separately
            x_coeffs = np.polyfit(t, x_coords, 2)
            y_coeffs = np.polyfit(t, y_coords, 2)
            
            # Calculate R² to assess fit quality
            x_pred = np.polyval(x_coeffs, t)
            y_pred = np.polyval(y_coeffs, t)
            
            x_r2 = 1 - np.sum((x_coords - x_pred)**2) / np.sum((x_coords - np.mean(x_coords))**2)
            y_r2 = 1 - np.sum((y_coords - y_pred)**2) / np.sum((y_coords - np.mean(y_coords))**2)
            
            confidence = (x_r2 + y_r2) / 2
            confidence = max(0.0, min(1.0, confidence))
            
            trajectory_params = {
                'x_coeffs': x_coeffs.tolist(),
                'y_coeffs': y_coeffs.tolist(),
                'start_time': len(positions)
            }
            
            return trajectory_params, confidence
            
        except Exception as e:
            logger.warning(f"Trajectory fitting failed: {e}")
            return {}, 0.0
    
    def _extrapolate_trajectory(self, 
                               trajectory_params: Dict, 
                               num_frames: int) -> List[Tuple[float, float]]:
        """Extrapolate trajectory into future frames"""
        future_positions = []
        
        try:
            x_coeffs = np.array(trajectory_params['x_coeffs'])
            y_coeffs = np.array(trajectory_params['y_coeffs'])
            start_time = trajectory_params['start_time']
            
            for i in range(1, num_frames + 1):
                t = start_time + i
                
                x = np.polyval(x_coeffs, t)
                y = np.polyval(y_coeffs, t)
                
                future_positions.append((float(x), float(y)))
        
        except Exception as e:
            logger.warning(f"Trajectory extrapolation failed: {e}")
        
        return future_positions
    
    def _estimate_ball_height(self, 
                             ball_detection: BallDetection, 
                             frame: np.ndarray) -> Optional[float]:
        """Estimate ball height above ground using shadow/size analysis"""
        try:
            # Simple height estimation based on ball size
            # (This would need calibration with actual field measurements)
            
            radius = ball_detection.radius
            
            # Assume ball appears smaller when higher
            # This is a simplified model - real implementation would need camera calibration
            
            if radius < 8:
                estimated_height = 50  # High in air
            elif radius < 12:
                estimated_height = 20  # Medium height
            elif radius < 16:
                estimated_height = 5   # Low
            else:
                estimated_height = 0   # On ground
            
            return estimated_height
            
        except Exception:
            return None
    
    def _predict_ball_position(self, 
                              frame: np.ndarray,
                              players: Optional[List[Detection]] = None,
                              frame_idx: int = 0) -> Optional[BallDetection]:
        """Predict ball position when not detected"""
        if not self.detection_history:
            return None
        
        try:
            # Use Kalman filter prediction
            predicted_pos = self.kalman_tracker.predict()
            
            if predicted_pos[0] <= 0 or predicted_pos[1] <= 0:
                return None
            
            # Get last known detection for reference
            last_detection = self.detection_history[-1]
            
            # Create predicted detection
            predicted_detection = BallDetection(
                position=predicted_pos,
                velocity=self.kalman_tracker.get_velocity(),
                acceleration=self.kalman_tracker.get_acceleration(),
                confidence=max(0.1, last_detection.confidence * 0.8),  # Decay confidence
                state=BallState.PREDICTED,
                radius=last_detection.radius,
                temporal_consistency=0.8
            )
            
            return predicted_detection
            
        except Exception as e:
            logger.warning(f"Ball prediction failed: {e}")
            return None
    
    def get_detection_statistics(self) -> Dict:
        """Get detection performance statistics"""
        total = max(1, self.detection_stats['total_frames'])
        
        return {
            'total_frames_processed': self.detection_stats['total_frames'],
            'successful_detection_rate': self.detection_stats['successful_detections'] / total,
            'yolo_detection_rate': self.detection_stats['yolo_detections'] / total,
            'circle_validation_rate': self.detection_stats['circle_validations'] / total,
            'template_match_rate': self.detection_stats['template_matches'] / total,
            'prediction_rate': self.detection_stats['predicted_positions'] / total,
            'current_state': self.current_ball_state.value,
            'frames_since_detection': self.frames_since_detection
        }
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.kalman_tracker = KalmanBallTracker()
        self.frame_history.clear()
        self.detection_history.clear()
        self.optical_flow_points = None
        self.current_ball_state = BallState.UNCERTAIN
        self.frames_since_detection = 0
        self.frames_since_contact = 0
        
        logger.info("Ball tracking reset")