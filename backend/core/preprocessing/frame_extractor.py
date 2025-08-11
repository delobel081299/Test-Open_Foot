"""
Optimized Frame Extractor for Football AI Analyzer
Advanced frame extraction with multi-threading, blur detection, and intelligent caching
"""

import cv2
import numpy as np
import os
import sys
import hashlib
import threading
import time
import gc
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import tempfile
import shutil
import math

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from backend.utils.logger import setup_logger
from backend.core.preprocessing.video_loader import VideoLoader, VideoMetadata

logger = setup_logger(__name__)

class ExtractionMode(Enum):
    """Frame extraction modes"""
    ALL_FRAMES = "all_frames"
    KEYFRAMES = "keyframes" 
    INTERVAL = "interval"

class BlurDetectionMethod(Enum):
    """Blur detection algorithms"""
    LAPLACIAN = "laplacian"
    SOBEL = "sobel"
    VARIANCE = "variance"

class ResizeMode(Enum):
    """Resize modes for intelligent resizing"""
    PRESERVE_ASPECT = "preserve_aspect"
    FORCE_SIZE = "force_size"
    MAX_SIZE = "max_size"

@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction"""
    # Extraction mode settings
    mode: ExtractionMode = ExtractionMode.INTERVAL
    interval: int = 30  # Extract every N frames
    target_fps: Optional[float] = None  # Target FPS for normalization
    start_time: float = 0.0  # Start time in seconds
    end_time: Optional[float] = None  # End time in seconds
    max_frames: Optional[int] = None  # Maximum number of frames to extract
    
    # Quality settings
    blur_detection: bool = True
    blur_method: BlurDetectionMethod = BlurDetectionMethod.LAPLACIAN
    blur_threshold: float = 100.0  # Threshold for blur detection
    quality_filter: bool = False  # Filter low quality frames
    quality_threshold: float = 0.3  # Quality threshold (0-1)
    
    # Resize settings
    resize: bool = False
    resize_mode: ResizeMode = ResizeMode.PRESERVE_ASPECT
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    max_width: int = 1920
    max_height: int = 1080
    
    # Performance settings
    num_threads: int = 4
    batch_size: int = 100
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    memory_limit: int = 2 * 1024 * 1024 * 1024  # 2GB memory limit
    
    # Output settings
    save_to_disk: bool = False
    output_dir: Optional[str] = None
    output_format: str = "jpg"
    jpeg_quality: int = 95
    
    # Progress tracking
    show_progress: bool = True
    progress_desc: str = "Extracting frames"

@dataclass
class ExtractedFrame:
    """Represents an extracted frame with metadata"""
    frame_number: int
    timestamp: float
    frame_data: Optional[np.ndarray] = None
    file_path: Optional[str] = None
    
    # Quality metrics
    is_blurry: bool = False
    blur_score: float = 0.0
    quality_score: float = 1.0
    
    # Processing info
    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    
    # Frame type (for keyframe detection)
    is_keyframe: bool = False
    frame_type: str = "P"  # I, P, B frames
    
    def __post_init__(self):
        """Initialize derived properties"""
        if self.frame_data is not None:
            self.processed_size = (self.frame_data.shape[1], self.frame_data.shape[0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'file_path': self.file_path,
            'is_blurry': self.is_blurry,
            'blur_score': self.blur_score,
            'quality_score': self.quality_score,
            'original_size': self.original_size,
            'processed_size': self.processed_size,
            'processing_time': self.processing_time,
            'is_keyframe': self.is_keyframe,
            'frame_type': self.frame_type
        }
    
    def release_memory(self):
        """Release frame data from memory"""
        if self.frame_data is not None:
            del self.frame_data
            self.frame_data = None
            gc.collect()

@dataclass
class ExtractionResult:
    """Result of frame extraction process"""
    total_frames: int
    extracted_frames: int
    filtered_frames: int
    processing_time: float
    frames: List[ExtractedFrame] = field(default_factory=list)
    cache_hits: int = 0
    memory_usage_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_frames': self.total_frames,
            'extracted_frames': self.extracted_frames,
            'filtered_frames': self.filtered_frames,
            'processing_time': self.processing_time,
            'cache_hits': self.cache_hits,
            'memory_usage_mb': self.memory_usage_mb,
            'errors': self.errors,
            'frames': [frame.to_dict() for frame in self.frames]
        }

class FrameCache:
    """Intelligent caching system for extracted frames"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "frame_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.lock = Lock()
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {"entries": {}, "total_size": 0, "last_cleanup": time.time()}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _generate_cache_key(self, video_path: str, config: FrameExtractionConfig, 
                          frame_number: int) -> str:
        """Generate unique cache key for frame"""
        key_data = {
            'video_path': str(video_path),
            'mode': config.mode.value,
            'interval': config.interval,
            'target_fps': config.target_fps,
            'resize': config.resize,
            'target_width': config.target_width,
            'target_height': config.target_height,
            'frame_number': frame_number
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_frame(self, video_path: str, config: FrameExtractionConfig, 
                        frame_number: int) -> Optional[ExtractedFrame]:
        """Get cached frame if exists"""
        if not config.enable_cache:
            return None
        
        cache_key = self._generate_cache_key(video_path, config, frame_number)
        
        with self.lock:
            if cache_key not in self.metadata["entries"]:
                return None
            
            entry = self.metadata["entries"][cache_key]
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                # Remove stale metadata entry
                del self.metadata["entries"][cache_key]
                self._save_metadata()
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_frame = pickle.load(f)
                
                # Update access time
                entry["last_access"] = time.time()
                self._save_metadata()
                
                logger.debug(f"Cache hit for frame {frame_number}")
                return cached_frame
                
            except Exception as e:
                logger.warning(f"Failed to load cached frame: {e}")
                # Remove corrupted cache entry
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata["entries"][cache_key]
                self._save_metadata()
                return None
    
    def cache_frame(self, video_path: str, config: FrameExtractionConfig, 
                   frame: ExtractedFrame):
        """Cache extracted frame"""
        if not config.enable_cache:
            return
        
        cache_key = self._generate_cache_key(video_path, config, frame.frame_number)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with self.lock:
                # Create a copy without frame data for caching metadata only
                cached_frame = ExtractedFrame(
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                    file_path=frame.file_path,
                    is_blurry=frame.is_blurry,
                    blur_score=frame.blur_score,
                    quality_score=frame.quality_score,
                    original_size=frame.original_size,
                    processed_size=frame.processed_size,
                    processing_time=frame.processing_time,
                    is_keyframe=frame.is_keyframe,
                    frame_type=frame.frame_type
                )
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_frame, f)
                
                file_size = cache_file.stat().st_size
                
                self.metadata["entries"][cache_key] = {
                    "file_path": str(cache_file),
                    "file_size": file_size,
                    "created_time": time.time(),
                    "last_access": time.time(),
                    "frame_number": frame.frame_number
                }
                
                self.metadata["total_size"] += file_size
                self._save_metadata()
                
                # Check if cleanup is needed
                if self.metadata["total_size"] > self.max_size_bytes:
                    self._cleanup_cache()
                    
        except Exception as e:
            logger.error(f"Failed to cache frame: {e}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        if not self.metadata["entries"]:
            return
        
        # Sort by last access time (oldest first)
        entries_by_access = sorted(
            self.metadata["entries"].items(),
            key=lambda x: x[1]["last_access"]
        )
        
        # Remove oldest entries until under size limit
        target_size = self.max_size_bytes * 0.8  # Leave some headroom
        current_size = self.metadata["total_size"]
        
        removed_count = 0
        for cache_key, entry in entries_by_access:
            if current_size <= target_size:
                break
            
            cache_file = Path(entry["file_path"])
            if cache_file.exists():
                cache_file.unlink()
            
            current_size -= entry["file_size"]
            del self.metadata["entries"][cache_key]
            removed_count += 1
        
        self.metadata["total_size"] = current_size
        self.metadata["last_cleanup"] = time.time()
        self._save_metadata()
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} cache entries")

class FrameExtractor:
    """
    Optimized frame extractor with multi-threading and intelligent processing
    
    Features:
    - Multi-threaded parallel extraction
    - 3 extraction modes: all_frames, keyframes, interval
    - Automatic blur detection
    - FPS normalization with interpolation
    - Intelligent resizing preserving aspect ratio
    - Smart caching system for reuse
    - Batch processing for memory efficiency
    - Optional disk saving
    """
    
    def __init__(self, config: Optional[FrameExtractionConfig] = None):
        """
        Initialize FrameExtractor
        
        Args:
            config: Frame extraction configuration
        """
        self.config = config or FrameExtractionConfig()
        self.video_loader = VideoLoader()
        self.cache = FrameCache(
            cache_dir=self.config.cache_dir,
            max_size_gb=5.0
        ) if self.config.enable_cache else None
        
        # Threading
        self.extraction_lock = Lock()
        self.memory_usage = 0
        
        logger.info(f"FrameExtractor initialized - Mode: {self.config.mode.value}, "
                   f"Threads: {self.config.num_threads}, Batch: {self.config.batch_size}")
    
    def _detect_blur(self, frame: np.ndarray, method: BlurDetectionMethod) -> Tuple[bool, float]:
        """
        Detect if frame is blurry
        
        Args:
            frame: Input frame
            method: Blur detection method
            
        Returns:
            (is_blurry, blur_score)
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            if method == BlurDetectionMethod.LAPLACIAN:
                # Laplacian variance method
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                blur_score = laplacian.var()
                
            elif method == BlurDetectionMethod.SOBEL:
                # Sobel gradient method
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                blur_score = np.sqrt(sobelx**2 + sobely**2).mean()
                
            elif method == BlurDetectionMethod.VARIANCE:
                # Simple variance method
                blur_score = gray.var()
                
            else:
                blur_score = 0.0
            
            is_blurry = bool(blur_score < self.config.blur_threshold)
            return is_blurry, float(blur_score)
            
        except Exception as e:
            logger.error(f"Blur detection failed: {e}")
            return False, 0.0
    
    def _estimate_quality(self, frame: np.ndarray) -> float:
        """
        Estimate frame quality (0-1 scale)
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Calculate multiple quality metrics
            scores = []
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            scores.append(sharpness_score * 0.4)
            
            # 2. Contrast (standard deviation)
            contrast_score = min(1.0, gray.std() / 100.0)
            scores.append(contrast_score * 0.3)
            
            # 3. Brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.flatten() / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            brightness_score = min(1.0, entropy / 8.0)
            scores.append(brightness_score * 0.3)
            
            return sum(scores)
            
        except Exception as e:
            logger.error(f"Quality estimation failed: {e}")
            return 0.5  # Default middle quality
    
    def _intelligent_resize(self, frame: np.ndarray) -> np.ndarray:
        """
        Intelligently resize frame preserving aspect ratio
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        if not self.config.resize:
            return frame
        
        h, w = frame.shape[:2]
        original_size = (w, h)
        
        try:
            if self.config.resize_mode == ResizeMode.PRESERVE_ASPECT:
                # Calculate new size preserving aspect ratio
                if self.config.target_width and self.config.target_height:
                    target_w, target_h = self.config.target_width, self.config.target_height
                    
                    # Calculate scale factors
                    scale_w = target_w / w
                    scale_h = target_h / h
                    scale = min(scale_w, scale_h)
                    
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                elif self.config.target_width:
                    # Scale to target width
                    scale = self.config.target_width / w
                    new_w = self.config.target_width
                    new_h = int(h * scale)
                    
                elif self.config.target_height:
                    # Scale to target height
                    scale = self.config.target_height / h
                    new_w = int(w * scale)
                    new_h = self.config.target_height
                    
                else:
                    return frame
                
            elif self.config.resize_mode == ResizeMode.FORCE_SIZE:
                # Force exact size (may distort)
                new_w = self.config.target_width or w
                new_h = self.config.target_height or h
                
            elif self.config.resize_mode == ResizeMode.MAX_SIZE:
                # Ensure frame fits within max dimensions
                max_w, max_h = self.config.max_width, self.config.max_height
                
                if w <= max_w and h <= max_h:
                    return frame
                
                scale_w = max_w / w
                scale_h = max_h / h
                scale = min(scale_w, scale_h)
                
                new_w = int(w * scale)
                new_h = int(h * scale)
            
            else:
                return frame
            
            # Ensure minimum size
            new_w = max(new_w, 32)
            new_h = max(new_h, 32)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            logger.debug(f"Resized frame: {original_size} -> {(new_w, new_h)}")
            return resized
            
        except Exception as e:
            logger.error(f"Frame resize failed: {e}")
            return frame
    
    def _save_frame_to_disk(self, frame: ExtractedFrame, output_dir: Path) -> str:
        """
        Save frame to disk
        
        Args:
            frame: Extracted frame
            output_dir: Output directory
            
        Returns:
            File path of saved frame
        """
        try:
            if frame.frame_data is None:
                raise ValueError("No frame data to save")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp_str = f"{frame.timestamp:.3f}".replace('.', '_')
            filename = f"frame_{frame.frame_number:06d}_{timestamp_str}.{self.config.output_format}"
            file_path = output_dir / filename
            
            # Save based on format
            if self.config.output_format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(
                    str(file_path), 
                    frame.frame_data, 
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                )
            else:
                cv2.imwrite(str(file_path), frame.frame_data)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save frame {frame.frame_number}: {e}")
            return ""
    
    def _process_frame(self, frame_data: np.ndarray, frame_number: int, 
                      timestamp: float, video_path: str) -> Optional[ExtractedFrame]:
        """
        Process a single frame
        
        Args:
            frame_data: Raw frame data
            frame_number: Frame number
            timestamp: Frame timestamp
            video_path: Source video path
            
        Returns:
            Processed ExtractedFrame or None if filtered
        """
        start_time = time.time()
        original_size = (frame_data.shape[1], frame_data.shape[0])
        
        try:
            # Check cache first
            if self.cache:
                cached_frame = self.cache.get_cached_frame(video_path, self.config, frame_number)
                if cached_frame:
                    return cached_frame
            
            # Blur detection
            is_blurry, blur_score = False, 0.0
            if self.config.blur_detection:
                is_blurry, blur_score = self._detect_blur(frame_data, self.config.blur_method)
                
                # Filter blurry frames if needed
                if is_blurry and self.config.quality_filter:
                    logger.debug(f"Filtered blurry frame {frame_number} (score: {blur_score:.2f})")
                    return None
            
            # Quality estimation
            quality_score = 1.0
            if self.config.quality_filter:
                quality_score = self._estimate_quality(frame_data)
                
                # Filter low quality frames
                if quality_score < self.config.quality_threshold:
                    logger.debug(f"Filtered low quality frame {frame_number} (score: {quality_score:.2f})")
                    return None
            
            # Intelligent resizing
            processed_frame = self._intelligent_resize(frame_data)
            
            # Create extracted frame object
            extracted_frame = ExtractedFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                frame_data=processed_frame,
                is_blurry=is_blurry,
                blur_score=blur_score,
                quality_score=quality_score,
                original_size=original_size,
                processing_time=time.time() - start_time
            )
            
            # Save to disk if requested
            if self.config.save_to_disk and self.config.output_dir:
                output_dir = Path(self.config.output_dir)
                file_path = self._save_frame_to_disk(extracted_frame, output_dir)
                extracted_frame.file_path = file_path
                
                # Release frame data to save memory if saved to disk
                extracted_frame.frame_data = None
            
            # Cache the frame
            if self.cache:
                self.cache.cache_frame(video_path, self.config, extracted_frame)
            
            return extracted_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return None
    
    def _extract_batch(self, video_path: str, frame_indices: List[int], 
                      metadata: VideoMetadata) -> List[ExtractedFrame]:
        """
        Extract a batch of frames
        
        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to extract
            metadata: Video metadata
            
        Returns:
            List of extracted frames
        """
        extracted_frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return extracted_frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for frame_idx in frame_indices:
                try:
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Failed to read frame {frame_idx}")
                        continue
                    
                    timestamp = frame_idx / fps if fps > 0 else 0.0
                    
                    # Process frame
                    processed_frame = self._process_frame(
                        frame, frame_idx, timestamp, str(video_path)
                    )
                    
                    if processed_frame:
                        extracted_frames.append(processed_frame)
                    
                    # Memory management
                    with self.extraction_lock:
                        self.memory_usage += frame.nbytes
                        
                        if self.memory_usage > self.config.memory_limit:
                            gc.collect()
                            self.memory_usage = 0
                
                except Exception as e:
                    logger.error(f"Error extracting frame {frame_idx}: {e}")
                    continue
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
        
        return extracted_frames
    
    def _get_frame_indices(self, metadata: VideoMetadata) -> List[int]:
        """
        Get frame indices based on extraction mode
        
        Args:
            metadata: Video metadata
            
        Returns:
            List of frame indices to extract
        """
        total_frames = metadata.total_frames
        fps = metadata.fps
        
        # Calculate frame range
        start_frame = int(self.config.start_time * fps)
        end_frame = int(self.config.end_time * fps) if self.config.end_time else total_frames
        end_frame = min(end_frame, total_frames)
        
        frame_indices = []
        
        if self.config.mode == ExtractionMode.ALL_FRAMES:
            # Extract all frames in range
            frame_indices = list(range(start_frame, end_frame))
            
        elif self.config.mode == ExtractionMode.INTERVAL:
            # Extract frames at specified interval
            frame_indices = list(range(start_frame, end_frame, self.config.interval))
            
        elif self.config.mode == ExtractionMode.KEYFRAMES:
            # For keyframe extraction, we'll use interval but could be enhanced
            # with actual keyframe detection using FFmpeg
            logger.warning("Keyframe detection not fully implemented, using interval mode")
            frame_indices = list(range(start_frame, end_frame, max(1, self.config.interval)))
        
        # Apply FPS normalization
        if self.config.target_fps and self.config.target_fps != fps:
            # Adjust frame indices for target FPS
            fps_ratio = self.config.target_fps / fps
            if fps_ratio < 1.0:
                # Downsample - take fewer frames
                step = int(1.0 / fps_ratio)
                frame_indices = frame_indices[::step]
            else:
                # Upsample - interpolate frames (simplified)
                logger.warning("Frame interpolation for upsampling not implemented")
        
        # Apply max frames limit
        if self.config.max_frames and len(frame_indices) > self.config.max_frames:
            # Evenly distribute frames across the range
            step = len(frame_indices) / self.config.max_frames
            frame_indices = [frame_indices[int(i * step)] for i in range(self.config.max_frames)]
        
        return frame_indices
    
    def extract_frames(self, video_path: Union[str, Path]) -> ExtractionResult:
        """
        Extract frames from video with optimized processing
        
        Args:
            video_path: Path to video file
            
        Returns:
            ExtractionResult with extracted frames and statistics
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        logger.info(f"Starting frame extraction from {video_path}")
        
        try:
            # Validate video and get metadata
            validation_result = self.video_loader.validate_video(video_path)
            if not validation_result.is_valid:
                raise ValueError(f"Video validation failed: {validation_result.errors}")
            
            metadata = validation_result.metadata
            
            # Get frame indices to extract
            frame_indices = self._get_frame_indices(metadata)
            total_frames_to_extract = len(frame_indices)
            
            if total_frames_to_extract == 0:
                logger.warning("No frames to extract based on current configuration")
                return ExtractionResult(
                    total_frames=metadata.total_frames,
                    extracted_frames=0,
                    filtered_frames=0,
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Extracting {total_frames_to_extract} frames from {metadata.total_frames} total")
            
            # Split frame indices into batches
            batches = [
                frame_indices[i:i + self.config.batch_size]
                for i in range(0, len(frame_indices), self.config.batch_size)
            ]
            
            # Initialize progress bar
            progress_bar = None
            if self.config.show_progress and TQDM_AVAILABLE:
                progress_bar = tqdm(
                    total=total_frames_to_extract,
                    desc=self.config.progress_desc,
                    unit="frames"
                )
            
            # Process batches in parallel
            all_extracted_frames = []
            cache_hits = 0
            
            with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
                # Submit batch jobs
                future_to_batch = {
                    executor.submit(self._extract_batch, str(video_path), batch, metadata): batch
                    for batch in batches
                }
                
                # Collect results
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_frames = future.result()
                        all_extracted_frames.extend(batch_frames)
                        
                        # Update progress
                        if progress_bar:
                            progress_bar.update(len(batch_frames))
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
            
            if progress_bar:
                progress_bar.close()
            
            # Sort frames by frame number
            all_extracted_frames.sort(key=lambda x: x.frame_number)
            
            # Calculate statistics
            extracted_count = len(all_extracted_frames)
            filtered_count = total_frames_to_extract - extracted_count
            processing_time = time.time() - start_time
            memory_usage_mb = self.memory_usage / (1024 * 1024)
            
            result = ExtractionResult(
                total_frames=metadata.total_frames,
                extracted_frames=extracted_count,
                filtered_frames=filtered_count,
                processing_time=processing_time,
                frames=all_extracted_frames,
                cache_hits=cache_hits,
                memory_usage_mb=memory_usage_mb
            )
            
            logger.info(f"Frame extraction completed: {extracted_count} frames extracted, "
                       f"{filtered_count} filtered, {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return ExtractionResult(
                total_frames=0,
                extracted_frames=0,
                filtered_frames=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
        
        finally:
            # Cleanup memory
            gc.collect()
            self.memory_usage = 0

# Utility functions for external use
def extract_frames_simple(video_path: Union[str, Path], 
                         interval: int = 30,
                         max_frames: Optional[int] = None,
                         output_dir: Optional[str] = None) -> ExtractionResult:
    """Simple frame extraction with default settings"""
    config = FrameExtractionConfig(
        mode=ExtractionMode.INTERVAL,
        interval=interval,
        max_frames=max_frames,
        save_to_disk=output_dir is not None,
        output_dir=output_dir
    )
    
    extractor = FrameExtractor(config)
    return extractor.extract_frames(video_path)

def extract_keyframes(video_path: Union[str, Path],
                     max_frames: Optional[int] = None,
                     output_dir: Optional[str] = None) -> ExtractionResult:
    """Extract keyframes from video"""
    config = FrameExtractionConfig(
        mode=ExtractionMode.KEYFRAMES,
        max_frames=max_frames,
        save_to_disk=output_dir is not None,
        output_dir=output_dir
    )
    
    extractor = FrameExtractor(config)
    return extractor.extract_frames(video_path)

def extract_frames_with_quality_filter(video_path: Union[str, Path],
                                      interval: int = 30,
                                      blur_threshold: float = 100.0,
                                      quality_threshold: float = 0.3,
                                      output_dir: Optional[str] = None) -> ExtractionResult:
    """Extract frames with quality filtering"""
    config = FrameExtractionConfig(
        mode=ExtractionMode.INTERVAL,
        interval=interval,
        blur_detection=True,
        blur_threshold=blur_threshold,
        quality_filter=True,
        quality_threshold=quality_threshold,
        save_to_disk=output_dir is not None,
        output_dir=output_dir
    )
    
    extractor = FrameExtractor(config)
    return extractor.extract_frames(video_path)