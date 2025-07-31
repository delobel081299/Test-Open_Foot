"""
Pytest configuration and fixtures for Football AI Analyzer tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.main import app
from backend.database.models import Base
from backend.database.session import get_db
from backend.utils.config import settings

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_db):
    """Create database session for testing"""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    """Create test client"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides = {}

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_video_path(temp_dir):
    """Create a sample video file for testing"""
    video_path = temp_dir / "test_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Generate 90 frames (3 seconds at 30fps)
    for i in range(90):
        # Create a frame with moving rectangle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add moving rectangle to simulate player
        x = int((i * 5) % 600)
        y = 200
        cv2.rectangle(frame, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
        
        # Add ball
        ball_x = int((i * 3) % 620)
        ball_y = 300
        cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 255), -1)
        
        out.write(frame)
    
    out.release()
    return video_path

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a 640x480 image with a person-like shape
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple person shape
    # Head
    cv2.circle(image, (320, 100), 30, (255, 255, 255), -1)
    
    # Body
    cv2.rectangle(image, (300, 130), (340, 250), (255, 255, 255), -1)
    
    # Arms
    cv2.rectangle(image, (270, 140), (300, 160), (255, 255, 255), -1)
    cv2.rectangle(image, (340, 140), (370, 160), (255, 255, 255), -1)
    
    # Legs
    cv2.rectangle(image, (305, 250), (320, 350), (255, 255, 255), -1)
    cv2.rectangle(image, (320, 250), (335, 350), (255, 255, 255), -1)
    
    return image

@pytest.fixture
def sample_detections():
    """Create sample detection data"""
    from backend.core.detection.yolo_detector import Detection
    
    detections = [
        Detection(
            bbox=(100, 100, 150, 200),
            confidence=0.85,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(200, 150, 250, 250),
            confidence=0.90,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(300, 180, 320, 200),
            confidence=0.75,
            class_id=32,
            class_name="sports ball"
        )
    ]
    
    return detections

@pytest.fixture
def sample_tracks():
    """Create sample tracking data"""
    from backend.core.tracking.byte_tracker import Track
    from backend.core.detection.yolo_detector import Detection
    
    # Create track 1
    track1 = Track(track_id=1)
    for i in range(10):
        detection = Detection(
            bbox=(100 + i*5, 100, 150 + i*5, 200),
            confidence=0.8 + i*0.01,
            class_id=0,
            class_name="person"
        )
        track1.update(detection, i)
    
    # Create track 2
    track2 = Track(track_id=2)
    for i in range(8):
        detection = Detection(
            bbox=(200 - i*3, 150, 250 - i*3, 250),
            confidence=0.85 + i*0.005,
            class_id=0,
            class_name="person"
        )
        track2.update(detection, i)
    
    return [track1, track2]

@pytest.fixture
def sample_poses():
    """Create sample pose data"""
    from backend.core.biomechanics.pose_extractor import Pose3D
    
    # Create keypoints for MediaPipe pose (33 keypoints)
    keypoints = np.random.rand(33, 4)  # x, y, z, visibility
    keypoints[:, 3] = np.random.uniform(0.5, 1.0, 33)  # Ensure visibility > 0.5
    
    pose = Pose3D(
        keypoints=keypoints,
        confidence=0.8,
        track_id=1,
        frame_number=0
    )
    
    return [pose]

@pytest.fixture
def mock_models(monkeypatch):
    """Mock AI models to avoid loading actual model files"""
    
    class MockYOLODetector:
        def __init__(self, *args, **kwargs):
            pass
            
        def detect(self, frame):
            from backend.core.detection.yolo_detector import Detection
            return [
                Detection(
                    bbox=(100, 100, 150, 200),
                    confidence=0.8,
                    class_id=0,
                    class_name="person"
                )
            ]
            
        def batch_detect(self, frames):
            return [self.detect(frame) for frame in frames]
    
    class MockPoseExtractor:
        def __init__(self, *args, **kwargs):
            pass
            
        def extract_pose(self, image, track_id, frame_number):
            from backend.core.biomechanics.pose_extractor import Pose3D
            keypoints = np.random.rand(33, 4)
            keypoints[:, 3] = 0.8  # visibility
            return Pose3D(keypoints, 0.8, track_id, frame_number)
    
    class MockByteTracker:
        def __init__(self, *args, **kwargs):
            self.tracks = {}
            self.next_id = 1
            
        def update(self, detections):
            from backend.core.tracking.byte_tracker import Track
            tracks = []
            for det in detections:
                if self.next_id not in self.tracks:
                    self.tracks[self.next_id] = Track(self.next_id)
                track = self.tracks[self.next_id]
                track.update(det, 0)
                tracks.append(track)
                self.next_id += 1
            return tracks
    
    # Patch the imports
    monkeypatch.setattr("backend.core.detection.yolo_detector.YOLODetector", MockYOLODetector)
    monkeypatch.setattr("backend.core.biomechanics.pose_extractor.PoseExtractor", MockPoseExtractor)
    monkeypatch.setattr("backend.core.tracking.byte_tracker.ByteTracker", MockByteTracker)

@pytest.fixture
def test_config():
    """Test configuration settings"""
    return {
        "gpu_enabled": False,  # Use CPU for tests
        "batch_size": 1,
        "confidence_threshold": 0.5,
        "test_mode": True
    }

@pytest.fixture(autouse=True)
def setup_test_dirs(temp_dir, monkeypatch):
    """Setup test directories"""
    # Create test directories
    (temp_dir / "uploads").mkdir()
    (temp_dir / "processed").mkdir()
    (temp_dir / "cache").mkdir()
    (temp_dir / "reports").mkdir()
    (temp_dir / "logs").mkdir()
    
    # Patch settings to use temp directories
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(temp_dir / "uploads"))
    monkeypatch.setattr(settings, "PROCESSED_DIR", str(temp_dir / "processed"))
    monkeypatch.setattr(settings, "CACHE_DIR", str(temp_dir / "cache"))
    monkeypatch.setattr(settings, "REPORTS_DIR", str(temp_dir / "reports"))
    monkeypatch.setattr(settings, "LOGS_DIR", str(temp_dir / "logs"))

# Utility functions for tests
def create_test_video_file(path: Path, duration: int = 3, fps: int = 30, resolution: tuple = (640, 480)):
    """Create a test video file"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, resolution)
    
    total_frames = duration * fps
    for i in range(total_frames):
        frame = np.random.randint(0, 255, (resolution[1], resolution[0], 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return path

def assert_video_properties(video_path: Path, min_duration: float = 1.0):
    """Assert basic video properties"""
    assert video_path.exists(), f"Video file does not exist: {video_path}"
    assert video_path.stat().st_size > 0, "Video file is empty"
    
    # Check with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Cannot open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    assert duration >= min_duration, f"Video too short: {duration}s < {min_duration}s"
    assert fps > 0, "Invalid FPS"

# Marks for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu