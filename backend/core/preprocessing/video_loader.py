import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ffmpeg
from dataclasses import dataclass

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    bitrate: int

class VideoLoader:
    """Handles video loading and frame extraction"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using ffmpeg"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse frame rate
            fps_str = video_stream['r_frame_rate']
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den
            else:
                fps = float(fps_str)
            
            # Calculate duration and total frames
            duration = float(probe['format']['duration'])
            total_frames = int(video_stream.get('nb_frames', fps * duration))
            
            return VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=video_stream['codec_name'],
                bitrate=int(probe['format'].get('bit_rate', 0))
            )
            
        except Exception as e:
            logger.error(f"Failed to get video metadata: {str(e)}")
            raise
    
    def validate_video(self, video_path: str) -> Dict[str, any]:
        """Validate video file and return info"""
        path = Path(video_path)
        
        # Check file exists
        if not path.exists():
            return {"valid": False, "error": "File not found"}
        
        # Check file extension
        if path.suffix.lower() not in self.supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
            }
        
        try:
            metadata = self.get_video_metadata(video_path)
            
            # Validate video properties
            if metadata.width < 640 or metadata.height < 480:
                return {
                    "valid": False,
                    "error": "Video resolution too low. Minimum: 640x480"
                }
            
            if metadata.fps < 15:
                return {
                    "valid": False,
                    "error": "Frame rate too low. Minimum: 15 FPS"
                }
            
            return {
                "valid": True,
                "metadata": metadata,
                "duration": metadata.duration,
                "fps": metadata.fps,
                "resolution": f"{metadata.width}x{metadata.height}"
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def extract_frames(
        self,
        video_path: str,
        fps: Optional[float] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[np.ndarray]:
        """Extract frames from video"""
        
        logger.info(f"Extracting frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        if fps is None:
            fps = video_fps
        frame_interval = int(video_fps / fps)
        
        # Calculate start and end frames
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps) if end_time else total_frames
        
        frames = []
        frame_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if current_frame > end_frame:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def extract_keyframes(self, video_path: str, threshold: float = 30.0) -> List[Tuple[int, np.ndarray]]:
        """Extract keyframes based on scene changes"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")
        
        keyframes = []
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(frame, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    keyframes.append((frame_idx, frame))
            else:
                # Always include first frame
                keyframes.append((frame_idx, frame))
            
            prev_frame = frame.copy()
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(keyframes)} keyframes")
        return keyframes
    
    def save_frames(self, frames: List[np.ndarray], output_dir: str, prefix: str = "frame"):
        """Save frames to disk"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            filename = output_path / f"{prefix}_{idx:06d}.jpg"
            cv2.imwrite(str(filename), frame)
        
        logger.info(f"Saved {len(frames)} frames to {output_dir}")
    
    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """Create video from frames"""
        
        if not frames:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Created video: {output_path}")