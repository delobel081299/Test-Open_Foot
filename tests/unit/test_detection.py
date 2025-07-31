"""
Unit tests for detection module
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from backend.core.detection.yolo_detector import YOLODetector, Detection


@pytest.mark.unit
class TestDetection:
    """Test Detection class"""
    
    def test_detection_creation(self):
        """Test Detection object creation"""
        detection = Detection(
            bbox=(10.0, 20.0, 50.0, 80.0),
            confidence=0.85,
            class_id=0,
            class_name="person"
        )
        
        assert detection.bbox == (10.0, 20.0, 50.0, 80.0)
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.track_id is None
    
    def test_detection_with_track_id(self):
        """Test Detection with track ID"""
        detection = Detection(
            bbox=(10.0, 20.0, 50.0, 80.0),
            confidence=0.85,
            class_id=0,
            class_name="person",
            track_id=5
        )
        
        assert detection.track_id == 5


@pytest.mark.unit
class TestYOLODetector:
    """Test YOLODetector class"""
    
    @patch('backend.core.detection.yolo_detector.YOLO')
    @patch('torch.cuda.is_available')
    def test_detector_initialization(self, mock_cuda, mock_yolo):
        """Test detector initialization"""
        mock_cuda.return_value = False
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector()
        
        assert detector.device == 'cpu'
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.4
        mock_yolo.assert_called_once()
    
    @patch('backend.core.detection.yolo_detector.YOLO')
    @patch('torch.cuda.is_available')
    def test_detector_gpu_initialization(self, mock_cuda, mock_yolo):
        """Test detector initialization with GPU"""
        mock_cuda.return_value = True
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        with patch('backend.utils.config.settings') as mock_settings:
            mock_settings.GPU_ENABLED = True
            detector = YOLODetector()
        
        assert detector.device == 'cuda'
    
    def test_confidence_adjustment(self, mock_models):
        """Test confidence threshold adjustment"""
        detector = YOLODetector()
        
        detector.adjust_confidence(0.7)
        assert detector.confidence_threshold == 0.7
        
        # Test bounds
        detector.adjust_confidence(0.05)  # Too low
        assert detector.confidence_threshold == 0.1
        
        detector.adjust_confidence(0.95)  # Too high
        assert detector.confidence_threshold == 0.9
    
    def test_detect_single_frame(self, mock_models, sample_image):
        """Test detection on single frame"""
        detector = YOLODetector()
        detections = detector.detect(sample_image)
        
        assert isinstance(detections, list)
        assert len(detections) >= 1
        
        for detection in detections:
            assert isinstance(detection, Detection)
            assert 0 <= detection.confidence <= 1
            assert len(detection.bbox) == 4
    
    def test_batch_detect(self, mock_models, sample_image):
        """Test batch detection"""
        detector = YOLODetector()
        frames = [sample_image, sample_image, sample_image]
        
        all_detections = detector.batch_detect(frames)
        
        assert len(all_detections) == 3
        for frame_detections in all_detections:
            assert isinstance(frame_detections, list)
    
    def test_filter_players(self, sample_detections):
        """Test player filtering"""
        detector = YOLODetector()
        players = detector.filter_players(sample_detections)
        
        # Should filter out non-person detections
        assert all(d.class_name == "person" for d in players)
        assert len(players) == 2  # From sample_detections fixture
    
    def test_filter_ball(self, sample_detections):
        """Test ball filtering"""
        detector = YOLODetector()
        ball = detector.filter_ball(sample_detections)
        
        assert ball is not None
        assert ball.class_name == "sports ball"
        assert ball.confidence == 0.75  # From sample_detections fixture
    
    def test_filter_ball_none(self):
        """Test ball filtering when no ball detected"""
        detector = YOLODetector()
        detections = [
            Detection((100, 100, 150, 200), 0.8, 0, "person")
        ]
        ball = detector.filter_ball(detections)
        
        assert ball is None
    
    def test_draw_detections(self, sample_detections, sample_image):
        """Test drawing detections on frame"""
        detector = YOLODetector()
        annotated = detector.draw_detections(sample_image, sample_detections)
        
        # Check that image was modified (not same as original)
        assert not np.array_equal(annotated, sample_image)
        assert annotated.shape == sample_image.shape
    
    def test_detection_stats(self, mock_models):
        """Test detection statistics calculation"""
        detector = YOLODetector()
        
        # Create mock detections for multiple frames
        detections_per_frame = [
            [
                Detection((100, 100, 150, 200), 0.8, 0, "person"),
                Detection((200, 150, 250, 250), 0.9, 0, "person"),
                Detection((300, 180, 320, 200), 0.7, 32, "sports ball")
            ],
            [
                Detection((110, 100, 160, 200), 0.85, 0, "person"),
                Detection((305, 185, 325, 205), 0.75, 32, "sports ball")
            ]
        ]
        
        stats = detector.get_detection_stats(detections_per_frame)
        
        assert stats["total_frames"] == 2
        assert stats["total_detections"] == 5
        assert stats["total_players"] == 3
        assert stats["total_balls"] == 2
        assert stats["avg_players_per_frame"] == 1.5
        assert 0.7 <= stats["avg_confidence"] <= 0.9


@pytest.mark.integration
class TestDetectionIntegration:
    """Integration tests for detection module"""
    
    def test_detection_pipeline_with_video(self, sample_video_path, mock_models):
        """Test detection pipeline with video file"""
        detector = YOLODetector()
        
        # Read video frames
        cap = cv2.VideoCapture(str(sample_video_path))
        frames = []
        
        for _ in range(10):  # Read first 10 frames
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        
        cap.release()
        
        # Run detection
        detections_per_frame = detector.batch_detect(frames)
        
        assert len(detections_per_frame) == len(frames)
        
        # Check that each frame has detections
        for frame_detections in detections_per_frame:
            assert isinstance(frame_detections, list)
            # Mock detector should return at least one detection
            assert len(frame_detections) >= 1
    
    def test_detection_with_empty_frame(self, mock_models):
        """Test detection with empty/black frame"""
        detector = YOLODetector()
        
        # Create black frame
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(black_frame)
        
        # Mock should still return detections
        assert isinstance(detections, list)
    
    def test_detection_performance(self, mock_models, sample_image):
        """Test detection performance (basic timing)"""
        import time
        
        detector = YOLODetector()
        
        start_time = time.time()
        for _ in range(10):
            detector.detect(sample_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Should be fast with mock model
        assert avg_time < 1.0  # Less than 1 second per detection