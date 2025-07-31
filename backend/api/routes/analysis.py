from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, Dict
from datetime import datetime
import asyncio

from backend.database.session import get_db
from backend.database import models, crud
from backend.core.preprocessing.video_loader import VideoLoader
from backend.core.detection.yolo_detector import YOLODetector
from backend.core.tracking.byte_tracker import ByteTracker
from backend.core.biomechanics.pose_extractor import PoseExtractor
from backend.core.technical.action_classifier import ActionClassifier
from backend.core.tactical.position_analyzer import PositionAnalyzer
from backend.core.scoring.score_aggregator import ScoreAggregator
from backend.utils.logger import setup_logger
from backend.utils.config import settings

router = APIRouter()
logger = setup_logger(__name__)

# Analysis pipeline components
video_loader = VideoLoader()
detector = YOLODetector()
tracker = ByteTracker()
pose_extractor = PoseExtractor()
action_classifier = ActionClassifier()
position_analyzer = PositionAnalyzer()
score_aggregator = ScoreAggregator()

async def run_analysis_pipeline(video_id: int, db: Session):
    """Run the complete analysis pipeline for a video"""
    
    try:
        # Update status
        crud.update_video_status(db, video_id, "processing")
        
        # Load video
        video = crud.get_video(db, video_id)
        video_path = f"data/uploads/{video.saved_filename}"
        
        logger.info(f"Starting analysis for video {video_id}")
        
        # 1. Preprocessing
        frames = await asyncio.to_thread(
            video_loader.extract_frames,
            video_path,
            fps=settings.ANALYSIS_FPS
        )
        
        # 2. Detection
        detections = await asyncio.to_thread(
            detector.batch_detect,
            frames
        )
        
        # 3. Tracking
        tracks = await asyncio.to_thread(
            tracker.process_video,
            detections
        )
        
        # 4. Biomechanics Analysis
        poses = await asyncio.to_thread(
            pose_extractor.extract_poses_from_tracks,
            tracks,
            frames
        )
        
        # 5. Technical Analysis
        actions = await asyncio.to_thread(
            action_classifier.classify_actions,
            tracks,
            frames
        )
        
        # 6. Tactical Analysis
        tactical_data = await asyncio.to_thread(
            position_analyzer.analyze_positions,
            tracks
        )
        
        # 7. Score Aggregation
        final_scores = await asyncio.to_thread(
            score_aggregator.calculate_scores,
            {
                "poses": poses,
                "actions": actions,
                "tactical": tactical_data
            }
        )
        
        # Save analysis results
        analysis = crud.create_analysis(
            db,
            video_id=video_id,
            scores=final_scores,
            detections_count=len(detections),
            tracks_count=len(tracks),
            actions_count=len(actions)
        )
        
        # Update video status
        crud.update_video_status(db, video_id, "completed")
        
        logger.info(f"Analysis completed for video {video_id}")
        
    except Exception as e:
        logger.error(f"Analysis failed for video {video_id}: {str(e)}")
        crud.update_video_status(db, video_id, "failed")
        raise

@router.post("/start/{video_id}")
async def start_analysis(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    analysis_params: Optional[Dict] = None
):
    """Start video analysis"""
    
    # Check if video exists
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if analysis already exists or in progress
    if video.status in ["processing", "completed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis already {video.status}"
        )
    
    # Start analysis in background
    background_tasks.add_task(run_analysis_pipeline, video_id, db)
    
    return {
        "video_id": video_id,
        "status": "started",
        "message": "Analysis started successfully"
    }

@router.get("/status/{video_id}")
async def get_analysis_status(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get analysis status for a video"""
    
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    analysis = crud.get_analysis_by_video(db, video_id)
    
    response = {
        "video_id": video_id,
        "status": video.status,
        "progress": 0
    }
    
    if analysis:
        response.update({
            "analysis_id": analysis.id,
            "completed_at": analysis.created_at,
            "detections_count": analysis.detections_count,
            "tracks_count": analysis.tracks_count,
            "actions_count": analysis.actions_count
        })
    
    return response

@router.post("/cancel/{video_id}")
async def cancel_analysis(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Cancel ongoing analysis"""
    
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status != "processing":
        raise HTTPException(
            status_code=400,
            detail="No analysis in progress"
        )
    
    # Update status
    crud.update_video_status(db, video_id, "cancelled")
    
    return {"message": "Analysis cancelled successfully"}