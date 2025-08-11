"""
Tests unitaires pour le module frame_extractor
"""

import unittest
import tempfile
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.core.preprocessing.frame_extractor import (
    FrameExtractor, FrameExtractionConfig, ExtractedFrame, ExtractionResult,
    ExtractionMode, BlurDetectionMethod, ResizeMode, FrameCache,
    extract_frames_simple, extract_keyframes, extract_frames_with_quality_filter
)

class TestFrameExtractionConfig(unittest.TestCase):
    """Test FrameExtractionConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = FrameExtractionConfig()
        
        self.assertEqual(config.mode, ExtractionMode.INTERVAL)
        self.assertEqual(config.interval, 30)
        self.assertTrue(config.blur_detection)
        self.assertEqual(config.blur_method, BlurDetectionMethod.LAPLACIAN)
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.batch_size, 100)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = FrameExtractionConfig(
            mode=ExtractionMode.ALL_FRAMES,
            blur_detection=False,
            resize=True,
            target_width=640,
            target_height=480
        )
        
        self.assertEqual(config.mode, ExtractionMode.ALL_FRAMES)
        self.assertFalse(config.blur_detection)
        self.assertTrue(config.resize)
        self.assertEqual(config.target_width, 640)
        self.assertEqual(config.target_height, 480)

class TestExtractedFrame(unittest.TestCase):
    """Test ExtractedFrame class"""
    
    def test_frame_creation(self):
        """Test frame creation"""
        frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        frame = ExtractedFrame(
            frame_number=100,
            timestamp=3.33,
            frame_data=frame_data,
            is_blurry=False,
            blur_score=150.5,
            quality_score=0.8
        )
        
        self.assertEqual(frame.frame_number, 100)
        self.assertEqual(frame.timestamp, 3.33)
        self.assertFalse(frame.is_blurry)
        self.assertEqual(frame.blur_score, 150.5)
        self.assertEqual(frame.quality_score, 0.8)
        self.assertEqual(frame.processed_size, (640, 480))
    
    def test_frame_to_dict(self):
        """Test frame serialization"""
        frame = ExtractedFrame(
            frame_number=50,
            timestamp=1.67,
            is_blurry=True,
            blur_score=50.0
        )
        
        frame_dict = frame.to_dict()
        
        self.assertIsInstance(frame_dict, dict)
        self.assertEqual(frame_dict['frame_number'], 50)
        self.assertEqual(frame_dict['timestamp'], 1.67)
        self.assertTrue(frame_dict['is_blurry'])
        self.assertEqual(frame_dict['blur_score'], 50.0)
    
    def test_memory_release(self):
        """Test memory release"""
        frame_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = ExtractedFrame(frame_number=1, timestamp=0.0, frame_data=frame_data)
        
        self.assertIsNotNone(frame.frame_data)
        frame.release_memory()
        self.assertIsNone(frame.frame_data)

class TestFrameCache(unittest.TestCase):
    """Test FrameCache class"""
    
    def setUp(self):
        """Set up test cache"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = FrameCache(cache_dir=self.temp_dir, max_size_gb=0.001)  # 1MB for testing
    
    def tearDown(self):
        """Clean up test cache"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_metadata(self):
        """Test cache metadata operations"""
        self.assertIsInstance(self.cache.metadata, dict)
        self.assertIn('entries', self.cache.metadata)
        self.assertIn('total_size', self.cache.metadata)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        config = FrameExtractionConfig()
        key1 = self.cache._generate_cache_key("video1.mp4", config, 100)
        key2 = self.cache._generate_cache_key("video1.mp4", config, 100)
        key3 = self.cache._generate_cache_key("video2.mp4", config, 100)
        
        self.assertEqual(key1, key2)  # Same inputs should generate same key
        self.assertNotEqual(key1, key3)  # Different inputs should generate different keys

class TestFrameExtractor(unittest.TestCase):
    """Test FrameExtractor class"""
    
    def setUp(self):
        """Set up test extractor"""
        self.config = FrameExtractionConfig(
            num_threads=2,
            batch_size=10,
            show_progress=False  # Disable progress bar for tests
        )
        self.extractor = FrameExtractor(self.config)
    
    def test_extractor_initialization(self):
        """Test extractor initialization"""
        self.assertIsNotNone(self.extractor.config)
        self.assertIsNotNone(self.extractor.video_loader)
        self.assertEqual(self.extractor.config.num_threads, 2)
        self.assertEqual(self.extractor.config.batch_size, 10)
    
    def test_blur_detection_laplacian(self):
        """Test Laplacian blur detection"""
        # Create sharp image
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[40:60, 40:60] = 255  # Sharp rectangle
        
        # Create blurry image
        blurry_image = cv2.GaussianBlur(sharp_image, (15, 15), 5)
        
        is_blurry_sharp, score_sharp = self.extractor._detect_blur(sharp_image, BlurDetectionMethod.LAPLACIAN)
        is_blurry_blur, score_blur = self.extractor._detect_blur(blurry_image, BlurDetectionMethod.LAPLACIAN)
        
        self.assertFalse(is_blurry_sharp)
        self.assertTrue(is_blurry_blur)
        self.assertGreater(score_sharp, score_blur)
    
    def test_blur_detection_sobel(self):
        """Test Sobel blur detection"""
        # Create image with edges
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, :50] = 255  # Vertical edge
        
        is_blurry, score = self.extractor._detect_blur(image, BlurDetectionMethod.SOBEL)
        
        self.assertIsInstance(is_blurry, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
    
    def test_quality_estimation(self):
        """Test quality estimation"""
        # Create high contrast image
        high_quality = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create low contrast image
        low_quality = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        quality_high = self.extractor._estimate_quality(high_quality)
        quality_low = self.extractor._estimate_quality(low_quality)
        
        self.assertGreaterEqual(quality_high, 0.0)
        self.assertLessEqual(quality_high, 1.0)
        self.assertGreaterEqual(quality_low, 0.0)
        self.assertLessEqual(quality_low, 1.0)
        self.assertGreater(quality_high, quality_low)
    
    def test_intelligent_resize_preserve_aspect(self):
        """Test intelligent resize with aspect ratio preservation"""
        original_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test resize to specific width
        config = FrameExtractionConfig(
            resize=True,
            resize_mode=ResizeMode.PRESERVE_ASPECT,
            target_width=320
        )
        extractor = FrameExtractor(config)
        
        resized = extractor._intelligent_resize(original_frame)
        
        self.assertEqual(resized.shape[1], 320)  # Width should be 320
        self.assertEqual(resized.shape[0], 240)  # Height should be proportional (320*480/640=240)
    
    def test_intelligent_resize_max_size(self):
        """Test intelligent resize with max size"""
        large_frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        
        config = FrameExtractionConfig(
            resize=True,
            resize_mode=ResizeMode.MAX_SIZE,
            max_width=1920,
            max_height=1080
        )
        extractor = FrameExtractor(config)
        
        resized = extractor._intelligent_resize(large_frame)
        
        self.assertLessEqual(resized.shape[1], 1920)  # Width should not exceed max
        self.assertLessEqual(resized.shape[0], 1080)  # Height should not exceed max
    
    def test_frame_indices_interval_mode(self):
        """Test frame indices generation for interval mode"""
        from backend.core.preprocessing.video_loader import VideoMetadata
        
        metadata = VideoMetadata(
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=900,  # 30 seconds at 30fps
            duration=30.0,
            codec='h264',
            bitrate=5000000
        )
        
        config = FrameExtractionConfig(
            mode=ExtractionMode.INTERVAL,
            interval=30,  # Every 30 frames (1 second)
            start_time=5.0,  # Start at 5 seconds
            end_time=15.0   # End at 15 seconds
        )
        extractor = FrameExtractor(config)
        
        indices = extractor._get_frame_indices(metadata)
        
        expected_start = int(5.0 * 30)  # Frame 150
        expected_end = int(15.0 * 30)   # Frame 450
        expected_indices = list(range(expected_start, expected_end, 30))
        
        self.assertEqual(indices, expected_indices)
    
    def test_frame_indices_all_frames_mode(self):
        """Test frame indices generation for all frames mode"""
        from backend.core.preprocessing.video_loader import VideoMetadata
        
        metadata = VideoMetadata(
            width=640,
            height=480,
            fps=25.0,
            total_frames=250,  # 10 seconds at 25fps
            duration=10.0,
            codec='h264',
            bitrate=2000000
        )
        
        config = FrameExtractionConfig(
            mode=ExtractionMode.ALL_FRAMES,
            max_frames=50  # Limit to 50 frames
        )
        extractor = FrameExtractor(config)
        
        indices = extractor._get_frame_indices(metadata)
        
        self.assertEqual(len(indices), 50)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 245)  # Should be evenly distributed

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_extract_frames_simple_config(self):
        """Test simple frame extraction configuration"""
        # This test only checks if the function creates proper config
        # Actual video processing would require a real video file
        
        # Test that function exists and can be called
        self.assertTrue(callable(extract_frames_simple))
        self.assertTrue(callable(extract_keyframes))
        self.assertTrue(callable(extract_frames_with_quality_filter))

if __name__ == '__main__':
    unittest.main()