"""
Complete Video Loading and Preprocessing Module
Handles video loading, validation, metadata extraction, and preprocessing
"""

import cv2
import numpy as np
import os
import sys
import hashlib
import threading
import time
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
import json
import tempfile
import shutil
import mimetypes

# FFmpeg imports
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class VideoOrientation(Enum):
    """Video orientation types"""
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"

class VideoQuality(Enum):
    """Video quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class VideoMetadata:
    """Complete video metadata information"""
    # Basic properties
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    
    # Technical details
    codec: str
    bitrate: int
    pixel_format: Optional[str] = None
    color_space: Optional[str] = None
    
    # File information
    file_size: int = 0
    file_path: str = ""
    file_hash: Optional[str] = None
    
    # Additional metadata
    orientation: VideoOrientation = VideoOrientation.LANDSCAPE
    aspect_ratio: float = field(init=False)
    quality_score: float = 0.0
    has_audio: bool = False
    audio_codec: Optional[str] = None
    creation_time: Optional[str] = None
    
    # Processing flags
    is_corrupted: bool = False
    corruption_details: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived properties"""
        if self.width > 0 and self.height > 0:
            self.aspect_ratio = self.width / self.height
            
            # Determine orientation
            if self.aspect_ratio > 1.1:
                self.orientation = VideoOrientation.LANDSCAPE
            elif self.aspect_ratio < 0.9:
                self.orientation = VideoOrientation.PORTRAIT
            else:
                self.orientation = VideoOrientation.SQUARE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'codec': self.codec,
            'bitrate': self.bitrate,
            'pixel_format': self.pixel_format,
            'color_space': self.color_space,
            'file_size': self.file_size,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'orientation': self.orientation.value,
            'aspect_ratio': self.aspect_ratio,
            'quality_score': self.quality_score,
            'has_audio': self.has_audio,
            'audio_codec': self.audio_codec,
            'creation_time': self.creation_time,
            'is_corrupted': self.is_corrupted,
            'corruption_details': self.corruption_details
        }

@dataclass
class ValidationResult:
    """Result of video validation process"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[VideoMetadata] = None
    validation_time: float = 0.0
    checks_performed: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
        logger.error(f"Validation error: {error}")
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)
        logger.warning(f"Validation warning: {warning}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'validation_time': self.validation_time,
            'checks_performed': self.checks_performed
        }

class VideoLoader:
    """
    Complete video loading and preprocessing system with advanced features
    
    Features:
    - Support for MP4, AVI, MOV, MKV formats
    - File size validation (2GB default limit)
    - FFmpeg-based corruption detection
    - Complete metadata extraction
    - Orientation detection
    - Quality estimation
    - Memory-efficient processing
    - Unicode path support
    """
    
    def __init__(self, 
                 max_file_size: int = 2 * 1024 * 1024 * 1024,  # 2GB default
                 enable_corruption_check: bool = True,
                 memory_limit: int = 1024 * 1024 * 1024,  # 1GB memory limit
                 temp_dir: Optional[str] = None):
        """
        Initialize VideoLoader
        
        Args:
            max_file_size: Maximum file size in bytes (default 2GB)
            enable_corruption_check: Enable FFmpeg corruption detection
            memory_limit: Memory limit for video processing
            temp_dir: Temporary directory for processing
        """
        self.max_file_size = max_file_size
        self.enable_corruption_check = enable_corruption_check
        self.memory_limit = memory_limit
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Supported formats
        self.supported_formats = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska'
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'resolution': {
                VideoQuality.LOW: (640, 480),
                VideoQuality.MEDIUM: (1280, 720),
                VideoQuality.HIGH: (1920, 1080),
                VideoQuality.ULTRA: (3840, 2160)
            },
            'bitrate': {
                VideoQuality.LOW: 1000000,      # 1 Mbps
                VideoQuality.MEDIUM: 5000000,   # 5 Mbps
                VideoQuality.HIGH: 10000000,    # 10 Mbps
                VideoQuality.ULTRA: 25000000    # 25 Mbps
            }
        }
        
        logger.info(f"VideoLoader initialized with max_file_size={max_file_size/1024/1024:.1f}MB")
    
    def _calculate_file_hash(self, file_path: Union[str, Path], chunk_size: int = 8192) -> str:
        """Calculate SHA-256 hash of video file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {str(e)}")
            return None
    
    def _validate_path(self, file_path: Union[str, Path]) -> Tuple[bool, Path, str]:
        """
        Validate file path with Unicode support
        
        Returns:
            (is_valid, path_obj, error_message)
        """
        try:
            # Convert to Path object for better Unicode handling
            if isinstance(file_path, str):
                # Ensure proper encoding for Unicode paths
                if sys.platform == "win32":
                    # Windows specific Unicode handling
                    path_obj = Path(file_path)
                else:
                    # Unix-like systems
                    path_obj = Path(file_path.encode('utf-8').decode('utf-8'))
            else:
                path_obj = Path(file_path)
            
            # Check if file exists
            if not path_obj.exists():
                return False, path_obj, f"File not found: {path_obj}"
            
            # Check if it's a file (not directory)
            if not path_obj.is_file():
                return False, path_obj, f"Path is not a file: {path_obj}"
            
            # Check file extension
            if path_obj.suffix.lower() not in self.supported_formats:
                supported = ', '.join(self.supported_formats.keys())
                return False, path_obj, f"Unsupported format '{path_obj.suffix}'. Supported: {supported}"
            
            return True, path_obj, ""
            
        except Exception as e:
            return False, Path(str(file_path)), f"Path validation error: {str(e)}"
    
    def _check_file_size(self, file_path: Path) -> Tuple[bool, int, str]:
        """Check if file size is within limits"""
        try:
            file_size = file_path.stat().st_size
            
            if file_size > self.max_file_size:
                size_mb = file_size / (1024 * 1024)
                limit_mb = self.max_file_size / (1024 * 1024)
                return False, file_size, f"File too large: {size_mb:.1f}MB (limit: {limit_mb:.1f}MB)"
            
            return True, file_size, ""
            
        except Exception as e:
            return False, 0, f"Failed to check file size: {str(e)}"
    
    def _extract_metadata_opencv(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata using OpenCV"""
        metadata = {}
        
        try:
            # Convert path to string with proper encoding
            path_str = str(file_path)
            
            cap = cv2.VideoCapture(path_str)
            if not cap.isOpened():
                raise ValueError("Failed to open video with OpenCV")
            
            # Basic properties
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['duration'] = metadata['total_frames'] / metadata['fps'] if metadata['fps'] > 0 else 0
            
            # Try to get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc:
                # Convert fourcc to string
                codec_bytes = [int(fourcc) & 0xFF,
                             int(fourcc >> 8) & 0xFF,
                             int(fourcc >> 16) & 0xFF,
                             int(fourcc >> 24) & 0xFF]
                metadata['codec'] = ''.join([chr(b) for b in codec_bytes if b != 0])
            else:
                metadata['codec'] = 'unknown'
            
            cap.release()
            
        except Exception as e:
            logger.error(f"OpenCV metadata extraction failed: {str(e)}")
            raise
        
        return metadata
    
    def _extract_metadata_ffmpeg(self, file_path: Path) -> Dict[str, Any]:
        """Extract detailed metadata using FFmpeg"""
        if not FFMPEG_AVAILABLE:
            raise ImportError("FFmpeg not available for metadata extraction")
        
        try:
            # Convert path to string with proper encoding
            path_str = str(file_path)
            
            # Probe video file
            probe = ffmpeg.probe(path_str)
            
            metadata = {}
            
            # Find video stream
            video_stream = None
            audio_stream = None
            
            for stream in probe['streams']:
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Extract video metadata
            metadata['width'] = int(video_stream['width'])
            metadata['height'] = int(video_stream['height'])
            metadata['codec'] = video_stream['codec_name']
            metadata['pixel_format'] = video_stream.get('pix_fmt', 'unknown')
            
            # Parse frame rate
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                metadata['fps'] = num / den if den != 0 else 30.0
            else:
                metadata['fps'] = float(fps_str)
            
            # Duration and frames
            if 'duration' in probe['format']:
                metadata['duration'] = float(probe['format']['duration'])
            else:
                metadata['duration'] = float(video_stream.get('duration', 0))
            
            metadata['total_frames'] = int(video_stream.get('nb_frames', 
                                         metadata['fps'] * metadata['duration']))
            
            # Bitrate
            metadata['bitrate'] = int(probe['format'].get('bit_rate', 0))
            
            # Color space
            metadata['color_space'] = video_stream.get('color_space', 'unknown')
            
            # Audio information
            metadata['has_audio'] = audio_stream is not None
            if audio_stream:
                metadata['audio_codec'] = audio_stream['codec_name']
            
            # Creation time
            if 'tags' in probe['format']:
                metadata['creation_time'] = probe['format']['tags'].get('creation_time')
            
            return metadata
            
        except Exception as e:
            logger.error(f"FFmpeg metadata extraction failed: {str(e)}")
            raise
    
    def _check_corruption_ffmpeg(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check for video corruption using FFmpeg"""
        if not FFMPEG_AVAILABLE or not self.enable_corruption_check:
            return False, []
        
        corruption_issues = []
        
        try:
            path_str = str(file_path)
            
            # Run FFmpeg with error detection
            process = (
                ffmpeg
                .input(path_str)
                .output('pipe:', format='null')
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                stderr_str = stderr.decode('utf-8', errors='ignore')
                
                # Check for common corruption indicators
                corruption_indicators = [
                    'Invalid data found',
                    'corrupt',
                    'truncated',
                    'error',
                    'invalid',
                    'damaged'
                ]
                
                for indicator in corruption_indicators:
                    if indicator.lower() in stderr_str.lower():
                        corruption_issues.append(f"FFmpeg detected: {indicator}")
                
                if not corruption_issues:
                    corruption_issues.append("FFmpeg processing failed (possible corruption)")
            
        except Exception as e:
            logger.warning(f"Corruption check failed: {str(e)}")
            corruption_issues.append(f"Corruption check error: {str(e)}")
        
        return len(corruption_issues) > 0, corruption_issues
    
    def _estimate_quality(self, metadata: Dict[str, Any]) -> float:
        """
        Estimate video quality score (0-100)
        
        Based on:
        - Resolution
        - Bitrate
        - Frame rate
        - Codec efficiency
        """
        try:
            score = 0.0
            
            # Resolution score (40% weight)
            width = metadata.get('width', 0)
            height = metadata.get('height', 0)
            pixels = width * height
            
            if pixels >= 3840 * 2160:  # 4K
                resolution_score = 100
            elif pixels >= 1920 * 1080:  # 1080p
                resolution_score = 80
            elif pixels >= 1280 * 720:  # 720p
                resolution_score = 60
            elif pixels >= 640 * 480:  # 480p
                resolution_score = 40
            else:
                resolution_score = 20
            
            score += resolution_score * 0.4
            
            # Bitrate score (30% weight)
            bitrate = metadata.get('bitrate', 0)
            if bitrate >= 25000000:  # 25 Mbps+
                bitrate_score = 100
            elif bitrate >= 10000000:  # 10 Mbps
                bitrate_score = 80
            elif bitrate >= 5000000:  # 5 Mbps
                bitrate_score = 60
            elif bitrate >= 1000000:  # 1 Mbps
                bitrate_score = 40
            else:
                bitrate_score = 20
            
            score += bitrate_score * 0.3
            
            # Frame rate score (20% weight)
            fps = metadata.get('fps', 0)
            if fps >= 60:
                fps_score = 100
            elif fps >= 30:
                fps_score = 80
            elif fps >= 24:
                fps_score = 60
            elif fps >= 15:
                fps_score = 40
            else:
                fps_score = 20
            
            score += fps_score * 0.2
            
            # Codec efficiency (10% weight)
            codec = metadata.get('codec', '').lower()
            codec_scores = {
                'h265': 100, 'hevc': 100, 'av1': 100,
                'h264': 80, 'avc': 80,
                'vp9': 75,
                'vp8': 60,
                'xvid': 50, 'divx': 50,
                'mpeg4': 40,
                'mpeg2': 30
            }
            
            codec_score = codec_scores.get(codec, 50)  # Default to 50 for unknown codecs
            score += codec_score * 0.1
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Quality estimation failed: {str(e)}")
            return 50.0  # Default middle score
    
    def extract_metadata(self, file_path: Union[str, Path], 
                        calculate_hash: bool = False) -> VideoMetadata:
        """
        Extract complete video metadata
        
        Args:
            file_path: Path to video file
            calculate_hash: Whether to calculate file hash (slower)
            
        Returns:
            VideoMetadata object with complete information
        """
        # Validate path
        is_valid, path_obj, error = self._validate_path(file_path)
        if not is_valid:
            raise ValueError(error)
        
        # Get file size
        file_size = path_obj.stat().st_size
        
        try:
            # Try FFmpeg first for most complete metadata
            if FFMPEG_AVAILABLE:
                metadata_dict = self._extract_metadata_ffmpeg(path_obj)
                logger.debug("Used FFmpeg for metadata extraction")
            else:
                metadata_dict = self._extract_metadata_opencv(path_obj)
                logger.debug("Used OpenCV for metadata extraction")
                
        except Exception as e:
            logger.warning(f"Primary metadata extraction failed: {str(e)}")
            # Fallback to OpenCV
            try:
                metadata_dict = self._extract_metadata_opencv(path_obj)
                logger.debug("Used OpenCV as fallback for metadata extraction")
            except Exception as e2:
                logger.error(f"All metadata extraction methods failed: {str(e2)}")
                raise ValueError(f"Could not extract metadata: {str(e2)}")
        
        # Check for corruption
        is_corrupted, corruption_details = self._check_corruption_ffmpeg(path_obj)
        
        # Calculate file hash if requested
        file_hash = None
        if calculate_hash:
            file_hash = self._calculate_file_hash(path_obj)
        
        # Create metadata object
        metadata = VideoMetadata(
            width=metadata_dict['width'],
            height=metadata_dict['height'],
            fps=metadata_dict['fps'],
            total_frames=metadata_dict['total_frames'],
            duration=metadata_dict['duration'],
            codec=metadata_dict['codec'],
            bitrate=metadata_dict.get('bitrate', 0),
            pixel_format=metadata_dict.get('pixel_format'),
            color_space=metadata_dict.get('color_space'),
            file_size=file_size,
            file_path=str(path_obj),
            file_hash=file_hash,
            has_audio=metadata_dict.get('has_audio', False),
            audio_codec=metadata_dict.get('audio_codec'),
            creation_time=metadata_dict.get('creation_time'),
            is_corrupted=is_corrupted,
            corruption_details=corruption_details
        )
        
        # Calculate quality score
        metadata.quality_score = self._estimate_quality(metadata_dict)
        
        logger.info(f"Extracted metadata for {path_obj}: {metadata.width}x{metadata.height}, "
                   f"{metadata.fps:.1f}fps, {metadata.duration:.1f}s, quality={metadata.quality_score:.1f}")
        
        return metadata
    
    def validate_video(self, file_path: Union[str, Path], 
                      strict_mode: bool = False) -> ValidationResult:
        """
        Comprehensive video validation
        
        Args:
            file_path: Path to video file
            strict_mode: Enable strict validation (more checks)
            
        Returns:
            ValidationResult with detailed validation information
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True)
        
        try:
            # 1. Path validation
            result.checks_performed.append("path_validation")
            is_valid, path_obj, error = self._validate_path(file_path)
            if not is_valid:
                result.add_error(error)
                return result
            
            # 2. File size validation
            result.checks_performed.append("file_size_validation")
            size_valid, file_size, size_error = self._check_file_size(path_obj)
            if not size_valid:
                result.add_error(size_error)
                return result
            
            # 3. Extract metadata
            result.checks_performed.append("metadata_extraction")
            try:
                metadata = self.extract_metadata(path_obj)
                result.metadata = metadata
            except Exception as e:
                result.add_error(f"Metadata extraction failed: {str(e)}")
                return result
            
            # 4. Basic validation checks
            result.checks_performed.append("basic_validation")
            
            # Check minimum resolution
            if metadata.width < 320 or metadata.height < 240:
                result.add_error(f"Resolution too low: {metadata.width}x{metadata.height} "
                               f"(minimum: 320x240)")
            
            # Check frame rate
            if metadata.fps < 1:
                result.add_error(f"Invalid frame rate: {metadata.fps}")
            elif metadata.fps < 15:
                result.add_warning(f"Low frame rate: {metadata.fps} fps")
            
            # Check duration
            if metadata.duration <= 0:
                result.add_error(f"Invalid duration: {metadata.duration}")
            elif metadata.duration < 1:
                result.add_warning(f"Very short video: {metadata.duration:.1f} seconds")
            
            # 5. Corruption check
            if metadata.is_corrupted:
                result.checks_performed.append("corruption_check")
                if strict_mode:
                    result.add_error("Video file appears to be corrupted")
                else:
                    result.add_warning("Possible video corruption detected")
            
            # 6. Strict mode additional checks
            if strict_mode:
                result.checks_performed.append("strict_validation")
                
                # Check for reasonable aspect ratio
                if metadata.aspect_ratio > 10 or metadata.aspect_ratio < 0.1:
                    result.add_warning(f"Unusual aspect ratio: {metadata.aspect_ratio:.2f}")
                
                # Check for reasonable bitrate
                if metadata.bitrate > 0:
                    if metadata.bitrate < 100000:  # Less than 100 kbps
                        result.add_warning(f"Very low bitrate: {metadata.bitrate/1000:.0f} kbps")
                    elif metadata.bitrate > 100000000:  # More than 100 Mbps
                        result.add_warning(f"Very high bitrate: {metadata.bitrate/1000000:.0f} Mbps")
                
                # Check quality score
                if metadata.quality_score < 30:
                    result.add_warning(f"Low quality score: {metadata.quality_score:.1f}")
            
            # 7. Memory estimation for processing
            result.checks_performed.append("memory_estimation")
            estimated_memory = self._estimate_memory_usage(metadata)
            if estimated_memory > self.memory_limit:
                result.add_warning(f"Video may require {estimated_memory/1024/1024:.0f}MB memory "
                                 f"(limit: {self.memory_limit/1024/1024:.0f}MB)")
            
        except Exception as e:
            result.add_error(f"Validation failed with exception: {str(e)}")
        
        finally:
            result.validation_time = time.time() - start_time
        
        logger.info(f"Validation completed in {result.validation_time:.2f}s: "
                   f"valid={result.is_valid}, errors={len(result.errors)}, "
                   f"warnings={len(result.warnings)}")
        
        return result
    
    def _estimate_memory_usage(self, metadata: VideoMetadata) -> int:
        """Estimate memory usage for video processing"""
        # Rough estimation: width * height * 3 (RGB) * typical buffer frames
        buffer_frames = min(30, metadata.fps)  # Buffer up to 1 second or fps
        bytes_per_frame = metadata.width * metadata.height * 3
        estimated_memory = int(bytes_per_frame * buffer_frames)
        return estimated_memory
    
    def extract_frames_generator(self, 
                                file_path: Union[str, Path],
                                fps: Optional[float] = None,
                                start_time: float = 0,
                                end_time: Optional[float] = None,
                                max_frames: Optional[int] = None,
                                target_size: Optional[Tuple[int, int]] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Memory-efficient frame extraction using generator
        
        Args:
            file_path: Path to video file
            fps: Target FPS for extraction (None = original)
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            max_frames: Maximum number of frames to extract
            target_size: Resize frames to (width, height)
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        # Validate path
        is_valid, path_obj, error = self._validate_path(file_path)
        if not is_valid:
            raise ValueError(error)
        
        path_str = str(path_obj)
        
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path_str}")
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling
            if fps is None:
                fps = video_fps
            
            frame_interval = max(1, int(video_fps / fps)) if fps > 0 else 1
            
            # Calculate frame range
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps) if end_time else total_frames
            end_frame = min(end_frame, total_frames)
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_extracted = 0
            current_frame_idx = start_frame
            
            logger.info(f"Starting frame extraction: start={start_frame}, end={end_frame}, "
                       f"interval={frame_interval}, target_fps={fps}")
            
            while current_frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should yield this frame
                if (current_frame_idx - start_frame) % frame_interval == 0:
                    # Resize if requested
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    
                    yield current_frame_idx, frame
                    frames_extracted += 1
                    
                    # Check max frames limit
                    if max_frames and frames_extracted >= max_frames:
                        break
                
                current_frame_idx += 1
                
                # Memory management - force garbage collection periodically
                if frames_extracted % 100 == 0:
                    gc.collect()
            
            logger.info(f"Extracted {frames_extracted} frames from {path_str}")
            
        finally:
            cap.release()
    
    def extract_frames_batch(self, 
                           file_path: Union[str, Path],
                           batch_size: int = 32,
                           **kwargs) -> Iterator[List[Tuple[int, np.ndarray]]]:
        """
        Extract frames in batches for efficient processing
        
        Args:
            file_path: Path to video file
            batch_size: Number of frames per batch
            **kwargs: Additional arguments for extract_frames_generator
            
        Yields:
            List of (frame_number, frame_array) tuples
        """
        batch = []
        
        for frame_idx, frame in self.extract_frames_generator(file_path, **kwargs):
            batch.append((frame_idx, frame))
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
                gc.collect()  # Memory management
        
        # Yield remaining frames
        if batch:
            yield batch
    
    def get_video_summary(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get a complete video summary including validation and metadata
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with complete video information
        """
        summary = {
            'file_path': str(file_path),
            'timestamp': time.time(),
            'processing_info': {
                'ffmpeg_available': FFMPEG_AVAILABLE,
                'opencv_version': cv2.__version__
            }
        }
        
        try:
            # Validate video
            validation_result = self.validate_video(file_path, strict_mode=True)
            summary['validation'] = validation_result.to_dict()
            
            # If validation passed, add additional analysis
            if validation_result.is_valid and validation_result.metadata:
                metadata = validation_result.metadata
                
                # Add quality analysis
                summary['quality_analysis'] = {
                    'overall_score': metadata.quality_score,
                    'resolution_category': self._get_resolution_category(metadata),
                    'bitrate_category': self._get_bitrate_category(metadata),
                    'orientation': metadata.orientation.value,
                    'aspect_ratio_standard': self._get_aspect_ratio_standard(metadata.aspect_ratio)
                }
                
                # Add processing recommendations
                summary['processing_recommendations'] = self._get_processing_recommendations(metadata)
            
        except Exception as e:
            summary['error'] = str(e)
            logger.error(f"Failed to generate video summary: {str(e)}")
        
        return summary
    
    def _get_resolution_category(self, metadata: VideoMetadata) -> str:
        """Get resolution category string"""
        pixels = metadata.width * metadata.height
        
        if pixels >= 3840 * 2160:
            return "4K (Ultra HD)"
        elif pixels >= 1920 * 1080:
            return "1080p (Full HD)"
        elif pixels >= 1280 * 720:
            return "720p (HD)"
        elif pixels >= 640 * 480:
            return "480p (SD)"
        else:
            return "Low Resolution"
    
    def _get_bitrate_category(self, metadata: VideoMetadata) -> str:
        """Get bitrate category string"""
        if metadata.bitrate == 0:
            return "Unknown"
        
        mbps = metadata.bitrate / 1000000
        
        if mbps >= 25:
            return f"Ultra High ({mbps:.1f} Mbps)"
        elif mbps >= 10:
            return f"High ({mbps:.1f} Mbps)"
        elif mbps >= 5:
            return f"Medium ({mbps:.1f} Mbps)"
        elif mbps >= 1:
            return f"Low ({mbps:.1f} Mbps)"
        else:
            return f"Very Low ({mbps:.1f} Mbps)"
    
    def _get_aspect_ratio_standard(self, aspect_ratio: float) -> str:
        """Get standard aspect ratio name"""
        # Common aspect ratios with tolerance
        ratios = {
            16/9: "16:9 (Widescreen)",
            4/3: "4:3 (Standard)",
            21/9: "21:9 (Ultrawide)",
            1/1: "1:1 (Square)",
            9/16: "9:16 (Vertical)"
        }
        
        tolerance = 0.1
        for ratio, name in ratios.items():
            if abs(aspect_ratio - ratio) < tolerance:
                return name
        
        return f"{aspect_ratio:.2f}:1 (Custom)"
    
    def _get_processing_recommendations(self, metadata: VideoMetadata) -> Dict[str, Any]:
        """Get processing recommendations based on video properties"""
        recommendations = {
            'suggested_fps': min(30, metadata.fps),
            'memory_efficient': metadata.width * metadata.height > 1920 * 1080,
            'resize_recommended': False,
            'quality_notes': []
        }
        
        # Resolution recommendations
        if metadata.width > 1920 or metadata.height > 1080:
            recommendations['resize_recommended'] = True
            recommendations['suggested_size'] = (1920, 1080)
            recommendations['quality_notes'].append("Consider resizing for faster processing")
        
        # Frame rate recommendations
        if metadata.fps > 30:
            recommendations['quality_notes'].append("High frame rate - consider reducing for processing")
        elif metadata.fps < 15:
            recommendations['quality_notes'].append("Low frame rate - may affect analysis quality")
        
        # Quality recommendations
        if metadata.quality_score < 50:
            recommendations['quality_notes'].append("Low quality video - results may be impacted")
        
        # Memory recommendations
        estimated_memory = self._estimate_memory_usage(metadata)
        if estimated_memory > self.memory_limit:
            recommendations['quality_notes'].append("Large video - use batch processing")
        
        return recommendations

# Utility functions for external use
def quick_validate(file_path: Union[str, Path], max_size_mb: int = 2048) -> bool:
    """Quick video validation - returns True if video is valid"""
    try:
        loader = VideoLoader(max_file_size=max_size_mb * 1024 * 1024)
        result = loader.validate_video(file_path)
        return result.is_valid
    except Exception:
        return False

def get_video_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Get basic video information - returns None if invalid"""
    try:
        loader = VideoLoader()
        metadata = loader.extract_metadata(file_path)
        return metadata.to_dict()
    except Exception:
        return None

def is_supported_format(file_path: Union[str, Path]) -> bool:
    """Check if video format is supported"""
    try:
        path_obj = Path(file_path)
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv'}
        return path_obj.suffix.lower() in supported_formats
    except Exception:
        return False