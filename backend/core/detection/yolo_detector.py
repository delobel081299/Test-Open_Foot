import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2

from backend.utils.logger import setup_logger
from backend.utils.config import settings

logger = setup_logger(__name__)

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

class YOLODetector:
    """High-precision YOLO detector for football analysis"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() and settings.GPU_ENABLED else 'cpu'
        self.model_path = model_path or "models/yolov10/yolov10x.pt"
        self.model = None
        self.class_names = {
            0: 'person',
            32: 'sports ball'
        }
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Set model to use FP16 for better performance on GPU
            if self.device == 'cuda':
                self.model.model.half()
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame"""
        
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=[0, 32],  # Only detect persons and sports balls
            verbose=False
        )
        
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
                        class_name=self.class_names.get(cls, 'unknown')
                    )
                    detections.append(detection)
        
        return detections
    
    def batch_detect(self, frames: List[np.ndarray], batch_size: int = 8) -> List[List[Detection]]:
        """Detect objects in multiple frames efficiently"""
        
        logger.info(f"Processing {len(frames)} frames in batches of {batch_size}")
        
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Run batch inference
            results = self.model(
                batch_frames,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                classes=[0, 32],
                verbose=False,
                stream=True
            )
            
            # Process results
            for r in results:
                frame_detections = []
                
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detection = Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(conf),
                            class_id=cls,
                            class_name=self.class_names.get(cls, 'unknown')
                        )
                        frame_detections.append(detection)
                
                all_detections.append(frame_detections)
        
        logger.info(f"Detected objects in {len(all_detections)} frames")
        return all_detections
    
    def filter_players(self, detections: List[Detection]) -> List[Detection]:
        """Filter to keep only player detections"""
        return [d for d in detections if d.class_name == 'person']
    
    def filter_ball(self, detections: List[Detection]) -> Optional[Detection]:
        """Find ball detection with highest confidence"""
        ball_detections = [d for d in detections if d.class_name == 'sports ball']
        
        if ball_detections:
            return max(ball_detections, key=lambda d: d.confidence)
        return None
    
    def adjust_confidence(self, threshold: float):
        """Adjust detection confidence threshold"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Choose color based on class
            if det.class_name == 'person':
                color = (0, 255, 0)  # Green for players
            elif det.class_name == 'sports ball':
                color = (0, 0, 255)  # Red for ball
            else:
                color = (255, 255, 255)  # White for others
            
            # Draw bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_frame
    
    def get_detection_stats(self, detections_per_frame: List[List[Detection]]) -> Dict:
        """Calculate detection statistics"""
        
        total_players = 0
        total_balls = 0
        avg_confidence = 0
        total_detections = 0
        
        for frame_detections in detections_per_frame:
            for det in frame_detections:
                total_detections += 1
                avg_confidence += det.confidence
                
                if det.class_name == 'person':
                    total_players += 1
                elif det.class_name == 'sports ball':
                    total_balls += 1
        
        avg_confidence = avg_confidence / total_detections if total_detections > 0 else 0
        
        return {
            "total_frames": len(detections_per_frame),
            "total_detections": total_detections,
            "total_players": total_players,
            "total_balls": total_balls,
            "avg_players_per_frame": total_players / len(detections_per_frame) if detections_per_frame else 0,
            "avg_confidence": avg_confidence,
            "detection_rate": total_detections / len(detections_per_frame) if detections_per_frame else 0
        }