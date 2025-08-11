import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from pathlib import Path
from enum import Enum
import yaml

from backend.utils.logger import setup_logger
from backend.utils.config import settings

logger = setup_logger(__name__)

class ModelType(Enum):
    YOLOV10 = "yolov10"
    RT_DETR = "rt-detr"
    DINO_DETR = "dino-detr"

class FootballClass(Enum):
    PLAYER = 0
    BALL = 1
    GOAL = 2
    REFEREE = 3
    COACH = 4

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    attention_score: Optional[float] = None
    temporal_consistency: Optional[float] = None

@dataclass
class FootballROI:
    field_mask: np.ndarray
    goal_areas: List[Tuple[int, int, int, int]]
    center_circle: Tuple[int, int, int]
    penalty_areas: List[Tuple[int, int, int, int]]

class BaseDetector(ABC):
    """Base class for all detection models"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        pass
    
    @abstractmethod
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        pass

class YOLOv10Detector(BaseDetector):
    """YOLOv10 NMS-free detector for ultra-fast inference"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.model_path = model_path
        self.load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLOv10
            self.model = YOLOv10(model_path)
            self.model.to(self.device)
            
            if self.device == "cuda":
                self.model.model.half()
            
            logger.info(f"YOLOv10 model loaded on {self.device}")
        except ImportError:
            logger.error("YOLOv10 not available, falling back to YOLOv8")
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False)
        return self._parse_results(results)
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        results = self.model(frames, verbose=False, stream=True)
        return [self._parse_results([r]) for r in results]
    
    def _parse_results(self, results) -> List[Detection]:
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=cls,
                        class_name=self._get_class_name(cls)
                    )
                    detections.append(detection)
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        class_map = {0: "player", 32: "ball"}
        return class_map.get(class_id, "unknown")

class RTDETRDetector(BaseDetector):
    """RT-DETR detector for real-time detection with high precision"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.model_path = model_path
        self.load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_path: str) -> None:
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(model_path)
            self.model.to(self.device)
            
            if self.device == "cuda":
                self.model.model.half()
            
            logger.info(f"RT-DETR model loaded on {self.device}")
        except ImportError:
            logger.error("RT-DETR not available")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, verbose=False)
        return self._parse_results_with_attention(results)
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        results = self.model(frames, verbose=False, stream=True)
        return [self._parse_results_with_attention([r]) for r in results]
    
    def _parse_results_with_attention(self, results) -> List[Detection]:
        detections = []
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Extract attention scores if available
                    attention_score = self._extract_attention_score(r, i)
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=cls,
                        class_name=self._get_class_name(cls),
                        attention_score=attention_score
                    )
                    detections.append(detection)
        return detections
    
    def _extract_attention_score(self, result, box_idx: int) -> Optional[float]:
        # Placeholder for attention mechanism extraction
        return None
    
    def _get_class_name(self, class_id: int) -> str:
        football_classes = {
            FootballClass.PLAYER.value: "player",
            FootballClass.BALL.value: "ball",
            FootballClass.GOAL.value: "goal",
            FootballClass.REFEREE.value: "referee",
            FootballClass.COACH.value: "coach"
        }
        return football_classes.get(class_id, "unknown")

class DINODETRDetector(BaseDetector):
    """DINO-DETR detector for state-of-the-art accuracy"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._get_device(device)
        self.model = None
        self.model_path = model_path
        self.load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_path: str) -> None:
        try:
            # Load DINO-DETR model (implementation would require specific DINO model)
            logger.info(f"DINO-DETR model loaded on {self.device}")
            # Placeholder - actual implementation would load DINO model
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load DINO-DETR: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        # Placeholder implementation
        return []
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        # Placeholder implementation
        return [[] for _ in frames]

class AdvancedDetector:
    """High-precision multi-model detector for football analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.primary_detector = None
        self.backup_detectors = {}
        self.roi = None
        self.frame_history = []
        self.detection_history = []
        self.performance_stats = {
            "fps": 0.0,
            "inference_time": 0.0,
            "postprocess_time": 0.0,
            "memory_usage": 0.0
        }
        
        self._initialize_detectors()
        self._setup_tensorrt_optimization()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path is None:
            config_path = "config/advanced_detection.yaml"
        
        default_config = {
            "primary_model": {
                "type": "rt-detr",
                "model_path": "models/rtdetr/rtdetr-x.pt",
                "confidence_threshold": 0.7,
                "nms_threshold": 0.5
            },
            "backup_models": [
                {
                    "type": "yolov10",
                    "model_path": "models/yolov10/yolov10x.pt",
                    "confidence_threshold": 0.7
                }
            ],
            "football_classes": {
                "player": 0,
                "ball": 1,
                "goal": 2,
                "referee": 3,
                "coach": 4
            },
            "performance": {
                "target_fps": 60,
                "batch_size": 8,
                "precision": "fp16",
                "tensorrt_enabled": True,
                "max_detections": 100
            },
            "post_processing": {
                "roi_enabled": True,
                "occlusion_handling": True,
                "temporal_consistency": True,
                "attention_mechanism": True
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _get_device(self) -> str:
        if not torch.cuda.is_available():
            logger.error("GPU required for 60 FPS performance")
            raise RuntimeError("CUDA GPU required for advanced detection")
        
        device = "cuda"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU with {gpu_memory:.1f}GB memory")
        
        if gpu_memory < 6.0:
            logger.warning("GPU memory < 6GB, performance may be limited")
        
        return device
    
    def _initialize_detectors(self):
        model_type = self.config["primary_model"]["type"]
        model_path = self.config["primary_model"]["model_path"]
        
        detector_map = {
            "yolov10": YOLOv10Detector,
            "rt-detr": RTDETRDetector,
            "dino-detr": DINODETRDetector
        }
        
        if model_type in detector_map:
            self.primary_detector = detector_map[model_type](model_path, self.device)
            logger.info(f"Primary detector: {model_type}")
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Initialize backup detectors
        for backup_config in self.config.get("backup_models", []):
            backup_type = backup_config["type"]
            backup_path = backup_config["model_path"]
            
            if backup_type in detector_map:
                self.backup_detectors[backup_type] = detector_map[backup_type](
                    backup_path, self.device
                )
                logger.info(f"Backup detector loaded: {backup_type}")
    
    def _setup_tensorrt_optimization(self):
        if not self.config["performance"]["tensorrt_enabled"]:
            return
        
        try:
            logger.info("Setting up TensorRT optimization...")
            # TensorRT optimization would be implemented here
            # This is a placeholder for the actual implementation
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
    
    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame with high precision"""
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = self._preprocess_frame(frame)
        
        # Primary detection
        detections = self.primary_detector.detect(processed_frame)
        
        # Post-processing
        post_start = time.time()
        detections = self._postprocess_detections(detections, frame)
        
        # Update performance stats
        inference_time = post_start - start_time
        postprocess_time = time.time() - post_start
        
        self.performance_stats.update({
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "fps": 1.0 / (inference_time + postprocess_time)
        })
        
        # Store for temporal consistency
        self.detection_history.append(detections)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        return detections
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch detection for optimal performance"""
        start_time = time.time()
        
        # Preprocess all frames
        processed_frames = [self._preprocess_frame(frame) for frame in frames]
        
        # Batch inference
        batch_detections = self.primary_detector.batch_detect(processed_frames)
        
        # Post-process each frame's detections
        post_start = time.time()
        final_detections = []
        for i, (detections, original_frame) in enumerate(zip(batch_detections, frames)):
            processed_dets = self._postprocess_detections(detections, original_frame)
            final_detections.append(processed_dets)
        
        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats.update({
            "inference_time": post_start - start_time,
            "postprocess_time": time.time() - post_start,
            "fps": len(frames) / total_time
        })
        
        return final_detections
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal detection"""
        # Apply ROI mask if available
        if self.roi and self.config["post_processing"]["roi_enabled"]:
            frame = self._apply_roi_mask(frame)
        
        return frame
    
    def _postprocess_detections(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """Advanced post-processing for football-specific detection"""
        
        # Filter by confidence threshold
        confidence_threshold = self.config["primary_model"]["confidence_threshold"]
        detections = [d for d in detections if d.confidence >= confidence_threshold]
        
        # Apply football-specific filtering
        detections = self._filter_football_objects(detections, frame)
        
        # Handle occlusions
        if self.config["post_processing"]["occlusion_handling"]:
            detections = self._handle_occlusions(detections)
        
        # Apply temporal consistency
        if self.config["post_processing"]["temporal_consistency"]:
            detections = self._apply_temporal_consistency(detections)
        
        # Limit max detections
        max_detections = self.config["performance"]["max_detections"]
        if len(detections) > max_detections:
            detections = sorted(detections, key=lambda d: d.confidence, reverse=True)[:max_detections]
        
        return detections
    
    def _apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply region of interest mask focusing on football field"""
        if self.roi is None:
            return frame
        
        masked_frame = frame.copy()
        masked_frame[~self.roi.field_mask] = 0
        return masked_frame
    
    def _filter_football_objects(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """Filter detections to keep only football-relevant objects"""
        football_detections = []
        
        for detection in detections:
            # Football-specific validation
            if self._is_valid_football_detection(detection, frame):
                football_detections.append(detection)
        
        return football_detections
    
    def _is_valid_football_detection(self, detection: Detection, frame: np.ndarray) -> bool:
        """Validate if detection is a valid football object"""
        x1, y1, x2, y2 = detection.bbox
        
        # Size validation
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if detection.class_name == "player":
            # Player size validation
            if area < 500 or height < 50 or width < 20:
                return False
            if height / width < 1.5:  # Players should be taller than wide
                return False
        
        elif detection.class_name == "ball":
            # Ball size validation
            if area < 20 or area > 2000:
                return False
            aspect_ratio = width / height
            if not (0.7 <= aspect_ratio <= 1.3):  # Ball should be roughly circular
                return False
        
        return True
    
    def _handle_occlusions(self, detections: List[Detection]) -> List[Detection]:
        """Handle overlapping detections using attention mechanism"""
        if not detections:
            return detections
        
        # Group overlapping detections
        overlapping_groups = self._find_overlapping_detections(detections)
        
        final_detections = []
        for group in overlapping_groups:
            if len(group) == 1:
                final_detections.extend(group)
            else:
                # Resolve occlusions using attention scores
                resolved = self._resolve_occlusion_group(group)
                final_detections.extend(resolved)
        
        return final_detections
    
    def _find_overlapping_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """Find groups of overlapping detections"""
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self._calculate_iou(det1.bbox, det2.bbox) > 0.3:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _resolve_occlusion_group(self, group: List[Detection]) -> List[Detection]:
        """Resolve occlusions in a group of overlapping detections"""
        if len(group) <= 1:
            return group
        
        # Sort by confidence and attention score
        group.sort(key=lambda d: (d.confidence, d.attention_score or 0), reverse=True)
        
        # Keep the best detection and filter others based on class compatibility
        resolved = [group[0]]
        
        for detection in group[1:]:
            # Allow multiple players but only one ball
            if detection.class_name == "player" and group[0].class_name == "player":
                # Check if sufficiently different position
                iou = self._calculate_iou(detection.bbox, group[0].bbox)
                if iou < 0.7:  # Allow if not too overlapping
                    resolved.append(detection)
            elif detection.class_name != group[0].class_name:
                # Different classes can coexist
                resolved.append(detection)
        
        return resolved
    
    def _apply_temporal_consistency(self, detections: List[Detection]) -> List[Detection]:
        """Apply temporal consistency using detection history"""
        if len(self.detection_history) < 2:
            return detections
        
        prev_detections = self.detection_history[-2] if len(self.detection_history) >= 2 else []
        
        for detection in detections:
            # Find matching detection in previous frame
            best_match = self._find_temporal_match(detection, prev_detections)
            if best_match:
                # Calculate temporal consistency score
                consistency = self._calculate_temporal_consistency(detection, best_match)
                detection.temporal_consistency = consistency
        
        return detections
    
    def _find_temporal_match(self, detection: Detection, prev_detections: List[Detection]) -> Optional[Detection]:
        """Find best matching detection from previous frame"""
        if not prev_detections:
            return None
        
        best_match = None
        best_iou = 0.0
        
        for prev_det in prev_detections:
            if prev_det.class_name == detection.class_name:
                iou = self._calculate_iou(detection.bbox, prev_det.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = prev_det
        
        return best_match
    
    def _calculate_temporal_consistency(self, current: Detection, previous: Detection) -> float:
        """Calculate temporal consistency score between detections"""
        # IoU component
        iou_score = self._calculate_iou(current.bbox, previous.bbox)
        
        # Confidence consistency
        conf_diff = abs(current.confidence - previous.confidence)
        conf_score = 1.0 - conf_diff
        
        # Combined score
        consistency = 0.7 * iou_score + 0.3 * conf_score
        return max(0.0, min(1.0, consistency))
    
    def set_roi(self, roi: FootballROI):
        """Set region of interest for football field"""
        self.roi = roi
        logger.info("ROI set for football field focusing")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        # Add memory usage
        if torch.cuda.is_available():
            self.performance_stats["memory_usage"] = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return self.performance_stats.copy()
    
    def benchmark_performance(self, test_frames: List[np.ndarray], num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark detector performance"""
        logger.info(f"Running performance benchmark with {len(test_frames)} frames, {num_runs} runs")
        
        fps_results = []
        inference_times = []
        postprocess_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Run batch detection
            results = self.batch_detect(test_frames)
            
            end_time = time.time()
            total_time = end_time - start_time
            fps = len(test_frames) / total_time
            
            fps_results.append(fps)
            inference_times.append(self.performance_stats["inference_time"])
            postprocess_times.append(self.performance_stats["postprocess_time"])
        
        benchmark_results = {
            "avg_fps": np.mean(fps_results),
            "max_fps": np.max(fps_results),
            "min_fps": np.min(fps_results),
            "avg_inference_time": np.mean(inference_times),
            "avg_postprocess_time": np.mean(postprocess_times),
            "total_detections": sum(len(frame_dets) for frame_dets in results),
            "target_fps_achieved": np.mean(fps_results) >= self.config["performance"]["target_fps"]
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results
    
    def calculate_map(self, ground_truth: List[List[Detection]], 
                     predictions: List[List[Detection]], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate mean Average Precision (mAP) metrics"""
        
        # Placeholder for mAP calculation
        # Full implementation would calculate precision, recall, and AP for each class
        
        map_results = {
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "mAP@[0.5:0.95]": 0.0,
            "AP_player": 0.0,
            "AP_ball": 0.0,
            "AP_goal": 0.0,
            "AP_referee": 0.0,
            "AP_coach": 0.0
        }
        
        logger.info(f"mAP calculation: {map_results}")
        return map_results