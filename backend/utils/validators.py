import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
try:
    import magic
except ImportError:
    # On Windows, use python-magic-bin
    import magic as magic
import ffmpeg
from fastapi import UploadFile

from backend.utils.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

def validate_video_file(file: UploadFile) -> Dict[str, Any]:
    """Validate uploaded video file"""
    
    try:
        # Check filename and extension
        if not file.filename:
            return {"valid": False, "error": "No filename provided"}
        
        file_path = Path(str(file.filename))
        extension = file_path.suffix.lower()
        
        # List of common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v']
        
        if extension not in video_extensions:
            return {
                "valid": False,
                "error": f"Unsupported format. Supported: {', '.join(video_extensions)}"
            }
        
        # Check file size if available
        size_mb = 0
        if hasattr(file, 'size') and file.size:
            size_mb = file.size / (1024 * 1024)
            if size_mb > 2000:  # 2GB limit
                return {
                    "valid": False,
                    "error": f"File too large. Maximum size: 2000MB"
                }
        
        # Basic validation passed
        return {
            "valid": True,
            "filename": file.filename,
            "extension": extension,
            "size_mb": size_mb
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}"
        }

def validate_video_content(file_path: str) -> Dict[str, Any]:
    """Validate video file content and extract metadata"""
    
    try:
        # Check if file exists
        if not Path(file_path).exists():
            return {"valid": False, "error": "File not found"}
        
        # Use ffprobe to get video information
        probe = ffmpeg.probe(file_path)
        
        # Find video stream
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        if not video_stream:
            return {"valid": False, "error": "No video stream found"}
        
        # Extract metadata
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Parse frame rate
        fps_str = video_stream['r_frame_rate']
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_str)
        
        duration = float(probe['format'].get('duration', 0))
        
        # Validate video properties
        if width < 640 or height < 480:
            return {
                "valid": False,
                "error": "Resolution too low. Minimum: 640x480"
            }
        
        if fps < 15:
            return {
                "valid": False,
                "error": "Frame rate too low. Minimum: 15 FPS"
            }
        
        if duration < 5:
            return {
                "valid": False,
                "error": "Video too short. Minimum: 5 seconds"
            }
        
        if duration > 3600:  # 1 hour
            return {
                "valid": False,
                "error": "Video too long. Maximum: 1 hour"
            }
        
        return {
            "valid": True,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "resolution": f"{width}x{height}",
            "codec": video_stream.get('codec_name', 'unknown'),
            "bitrate": int(probe['format'].get('bit_rate', 0))
        }
        
    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")
        return {"valid": False, "error": f"Video validation failed: {str(e)}"}

def validate_image_quality(frame: np.ndarray) -> Dict[str, Any]:
    """Validate image quality for analysis"""
    
    if frame is None or frame.size == 0:
        return {"valid": False, "error": "Empty frame"}
    
    # Check dimensions
    height, width = frame.shape[:2]
    if height < 480 or width < 640:
        return {
            "valid": False,
            "error": f"Frame resolution too low: {width}x{height}"
        }
    
    # Check if frame is too dark or too bright
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30:
        return {"valid": False, "error": "Frame too dark"}
    
    if mean_brightness > 225:
        return {"valid": False, "error": "Frame too bright"}
    
    # Check for blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return {"valid": False, "error": "Frame too blurry"}
    
    return {
        "valid": True,
        "brightness": mean_brightness,
        "sharpness": laplacian_var,
        "resolution": f"{width}x{height}"
    }

def validate_detection_bbox(bbox: tuple, image_shape: tuple) -> bool:
    """Validate bounding box coordinates"""
    
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
        return False
    
    # Check order
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Check minimum size
    min_size = 20
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False
    
    return True

def validate_tracking_confidence(confidence: float) -> bool:
    """Validate tracking confidence score"""
    return 0.0 <= confidence <= 1.0

def validate_pose_keypoints(keypoints: np.ndarray) -> Dict[str, Any]:
    """Validate pose keypoints array"""
    
    # Check shape (should be 33 keypoints x 4 coordinates)
    expected_shape = (33, 4)
    if keypoints.shape != expected_shape:
        return {
            "valid": False,
            "error": f"Invalid keypoints shape: {keypoints.shape}, expected: {expected_shape}"
        }
    
    # Check for valid coordinates (normalized 0-1 for x,y, any for z, 0-1 for visibility)
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    visibility = keypoints[:, 3]
    
    if np.any((x_coords < 0) | (x_coords > 1)):
        return {"valid": False, "error": "Invalid x coordinates (should be 0-1)"}
    
    if np.any((y_coords < 0) | (y_coords > 1)):
        return {"valid": False, "error": "Invalid y coordinates (should be 0-1)"}
    
    if np.any((visibility < 0) | (visibility > 1)):
        return {"valid": False, "error": "Invalid visibility scores (should be 0-1)"}
    
    # Check for minimum number of visible keypoints
    visible_count = np.sum(visibility > 0.5)
    min_visible = 10  # Minimum keypoints needed for analysis
    
    if visible_count < min_visible:
        return {
            "valid": False,
            "error": f"Too few visible keypoints: {visible_count}/{min_visible}"
        }
    
    return {
        "valid": True,
        "visible_keypoints": visible_count,
        "total_keypoints": len(keypoints),
        "visibility_score": np.mean(visibility)
    }

def validate_analysis_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate analysis parameters"""
    
    errors = []
    
    # Check confidence thresholds
    if 'detection_confidence' in params:
        conf = params['detection_confidence']
        if not isinstance(conf, (int, float)) or not (0.1 <= conf <= 0.9):
            errors.append("Detection confidence must be between 0.1 and 0.9")
    
    if 'tracking_confidence' in params:
        conf = params['tracking_confidence']
        if not isinstance(conf, (int, float)) or not (0.1 <= conf <= 0.9):
            errors.append("Tracking confidence must be between 0.1 and 0.9")
    
    # Check FPS
    if 'analysis_fps' in params:
        fps = params['analysis_fps']
        if not isinstance(fps, (int, float)) or not (5 <= fps <= 60):
            errors.append("Analysis FPS must be between 5 and 60")
    
    # Check batch size
    if 'batch_size' in params:
        batch_size = params['batch_size']
        if not isinstance(batch_size, int) or not (1 <= batch_size <= 32):
            errors.append("Batch size must be between 1 and 32")
    
    # Check max players
    if 'max_players' in params:
        max_players = params['max_players']
        if not isinstance(max_players, int) or not (2 <= max_players <= 30):
            errors.append("Max players must be between 2 and 30")
    
    if errors:
        return {"valid": False, "errors": errors}
    
    return {"valid": True}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    
    # Remove path separators and dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    
    clean_name = filename
    for char in dangerous_chars:
        clean_name = clean_name.replace(char, '_')
    
    # Limit length
    if len(clean_name) > 255:
        name_part = Path(clean_name).stem[:240]
        ext_part = Path(clean_name).suffix
        clean_name = f"{name_part}{ext_part}"
    
    return clean_name

def validate_file_permissions(file_path: str) -> bool:
    """Check if file has proper read permissions"""
    
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False