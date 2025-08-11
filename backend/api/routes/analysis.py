from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import asyncio
import json
import time
import uuid

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

# Job management
analysis_jobs: Dict[str, Dict[str, Any]] = {}
active_jobs: Dict[str, asyncio.Task] = {}

# Analysis pipeline components
video_loader = VideoLoader()
detector = YOLODetector()
tracker = ByteTracker()
pose_extractor = PoseExtractor()
try:
    action_classifier = ActionClassifier()
except Exception as e:
    logger.warning(f"Failed to load ActionClassifier: {e}")
    action_classifier = None
position_analyzer = PositionAnalyzer()
score_aggregator = ScoreAggregator()

def update_job_progress(job_id: str, stage: str, progress: int, message: str = None):
    """Update job progress"""
    if job_id in analysis_jobs:
        analysis_jobs[job_id].update({
            "current_stage": stage,
            "progress": progress,
            "message": message or f"Processing {stage}",
            "updated_at": time.time()
        })

async def run_analysis_pipeline(job_id: str, video_id: int, db: Session, config: Dict = None):
    """Run the complete analysis pipeline for a video with progress tracking"""
    
    try:
        # Initialize job tracking
        analysis_jobs[job_id] = {
            "video_id": video_id,
            "status": "processing",
            "current_stage": "initialization",
            "progress": 0,
            "started_at": time.time(),
            "message": "Initializing analysis pipeline",
            "config": config or {}
        }
        
        # Update database status
        crud.update_video_status(db, video_id, "processing")
        
        # Load video
        video = crud.get_video(db, video_id)
        video_path = f"data/uploads/{video.saved_filename}"
        
        logger.info(f"Starting analysis for job {job_id}, video {video_id}")
        
        # Stage 1: Preprocessing (0-15%)
        update_job_progress(job_id, "preprocessing", 5, "Extracting video frames")
        frames = await asyncio.to_thread(
            video_loader.extract_frames,
            video_path,
            fps=config.get("fps", getattr(settings, "ANALYSIS_FPS", 25))
        )
        update_job_progress(job_id, "preprocessing", 15, f"Extracted {len(frames)} frames")
        
        # Stage 2: Detection (15-35%)
        update_job_progress(job_id, "detection", 20, "Running object detection")
        detections = []
        for i, frame_batch in enumerate([frames[i:i+10] for i in range(0, len(frames), 10)]):
            batch_detections = await asyncio.to_thread(detector.batch_detect, frame_batch)
            detections.extend(batch_detections)
            progress = 20 + int((i * 10 / len(frames)) * 15)
            update_job_progress(job_id, "detection", progress)
        update_job_progress(job_id, "detection", 35, f"Detected {len(detections)} objects")
        
        # Stage 3: Tracking (35-50%)
        update_job_progress(job_id, "tracking", 40, "Tracking players across frames")
        tracks = await asyncio.to_thread(tracker.process_video, detections)
        update_job_progress(job_id, "tracking", 50, f"Generated {len(tracks)} player tracks")
        
        # Stage 4: Biomechanics Analysis (50-65%)
        update_job_progress(job_id, "biomechanics", 55, "Analyzing player movements")
        poses = await asyncio.to_thread(
            pose_extractor.extract_poses_from_tracks,
            tracks,
            frames
        )
        update_job_progress(job_id, "biomechanics", 65, "Movement analysis completed")
        
        # Stage 5: Technical Analysis (65-80%)
        if action_classifier:
            update_job_progress(job_id, "technical", 70, "Classifying football actions")
            actions = await asyncio.to_thread(
                action_classifier.classify_actions,
                tracks,
                frames
            )
            update_job_progress(job_id, "technical", 80, f"Classified {len(actions)} actions")
        else:
            actions = []
            update_job_progress(job_id, "technical", 80, "Action classification skipped")
        
        # Stage 6: Tactical Analysis (80-90%)
        update_job_progress(job_id, "tactical", 85, "Analyzing tactical formations")
        tactical_data = await asyncio.to_thread(position_analyzer.analyze_positions, tracks)
        update_job_progress(job_id, "tactical", 90, "Tactical analysis completed")
        
        # Stage 7: Score Aggregation (90-95%)
        update_job_progress(job_id, "aggregation", 92, "Calculating performance scores")
        final_scores = await asyncio.to_thread(
            score_aggregator.calculate_scores,
            {
                "poses": poses,
                "actions": actions,
                "tactical": tactical_data
            }
        )
        
        # Save analysis results (95-100%)
        update_job_progress(job_id, "saving", 97, "Saving analysis results")
        analysis = crud.create_analysis(
            db,
            video_id=video_id,
            scores=final_scores,
            detections_count=len(detections),
            tracks_count=len(tracks),
            actions_count=len(actions),
            metadata={"config": config, "job_id": job_id}
        )
        
        # Update final status
        analysis_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_stage": "completed",
            "message": "Analysis completed successfully",
            "completed_at": time.time(),
            "analysis_id": analysis.id,
            "results_summary": {
                "detections": len(detections),
                "tracks": len(tracks),
                "actions": len(actions),
                "final_score": final_scores.get("overall_score", 0)
            }
        })
        
        crud.update_video_status(db, video_id, "completed")
        logger.info(f"Analysis completed for job {job_id}")
        
    except asyncio.CancelledError:
        analysis_jobs[job_id].update({
            "status": "cancelled",
            "message": "Analysis was cancelled",
            "cancelled_at": time.time()
        })
        crud.update_video_status(db, video_id, "cancelled")
        logger.info(f"Analysis cancelled for job {job_id}")
        raise
    except Exception as e:
        logger.error(f"Analysis failed for job {job_id}: {str(e)}")
        analysis_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": time.time()
        })
        crud.update_video_status(db, video_id, "failed")
        raise

@router.post("/analyze/{job_id}",
            summary="Start Video Analysis",
            description="Launch analysis for uploaded video with customizable configuration")
async def analyze_video(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: Request,
    config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """
    Start comprehensive video analysis with advanced configuration:
    - Async processing with job queue management
    - Configurable analysis parameters
    - Real-time progress tracking
    - WebSocket notifications support
    """
    
    # Default configuration
    analysis_config = {
        "detection_confidence": 0.5,
        "tracking_max_age": 30,
        "fps": 25,
        "analyze_poses": True,
        "analyze_actions": True,
        "analyze_tactics": True,
        "generate_heatmaps": True,
        "detailed_scoring": True
    }
    
    # Update with provided config
    if config:
        analysis_config.update(config)
    
    # Find video by job_id (assuming job_id maps to a video)
    # This could be enhanced to support multiple videos per job
    from backend.api.routes.upload import upload_progress
    
    if job_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload job not found")
    
    upload_data = upload_progress[job_id]
    if "video_id" not in upload_data:
        raise HTTPException(status_code=400, detail="Upload not completed")
    
    video_id = upload_data["video_id"]
    
    # Check if video exists
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if analysis already exists or in progress
    if video.status in ["processing"]:
        # Check if there's already an active job
        existing_job = None
        for existing_job_id, job_data in analysis_jobs.items():
            if job_data.get("video_id") == video_id and job_data.get("status") == "processing":
                existing_job = existing_job_id
                break
        
        if existing_job:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Analysis already in progress",
                    "existing_job_id": existing_job
                }
            )
    
    # Check for completed analysis
    if video.status == "completed":
        raise HTTPException(
            status_code=400,
            detail="Analysis already completed. Use /results endpoint to get results."
        )
    
    # Generate new job ID for analysis if needed
    analysis_job_id = f"analysis_{job_id}"
    
    # Start analysis task
    task = asyncio.create_task(
        run_analysis_pipeline(analysis_job_id, video_id, db, analysis_config)
    )
    active_jobs[analysis_job_id] = task
    
    logger.info(f"Started analysis job {analysis_job_id} for video {video_id}")
    
    return {
        "job_id": analysis_job_id,
        "video_id": video_id,
        "status": "started",
        "message": "Analysis started successfully",
        "config": analysis_config,
        "progress_endpoint": f"/api/status/{analysis_job_id}",
        "results_endpoint": f"/api/results/{analysis_job_id}"
    }

async def generate_sse_events(job_id: str) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for job progress"""
    
    last_update = 0
    while True:
        if job_id not in analysis_jobs:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            break
        
        job_data = analysis_jobs[job_id]
        current_update = job_data.get("updated_at", 0)
        
        # Send update if there's new data
        if current_update > last_update:
            event_data = {
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "progress": job_data.get("progress", 0),
                "current_stage": job_data.get("current_stage", ""),
                "message": job_data.get("message", ""),
                "timestamp": current_update
            }
            
            # Add completion details if finished
            if job_data.get("status") in ["completed", "failed", "cancelled"]:
                if "results_summary" in job_data:
                    event_data["results_summary"] = job_data["results_summary"]
                if "error" in job_data:
                    event_data["error"] = job_data["error"]
                
                yield f"data: {json.dumps(event_data)}\n\n"
                break
            
            yield f"data: {json.dumps(event_data)}\n\n"
            last_update = current_update
        
        await asyncio.sleep(1)  # Poll every second

@router.get("/status/{job_id}",
           summary="Get Analysis Status",
           description="Get current status and progress of analysis job with SSE support")
async def get_analysis_status(
    job_id: str,
    request: Request,
    sse: bool = False
):
    """
    Get analysis status with optional Server-Sent Events streaming:
    - Real-time progress updates
    - Stage information with ETA
    - Error handling and completion notifications
    """
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    
    # Return SSE stream if requested
    if sse:
        return StreamingResponse(
            generate_sse_events(job_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    # Return current status
    job_data = analysis_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "video_id": job_data.get("video_id"),
        "status": job_data.get("status", "unknown"),
        "progress": job_data.get("progress", 0),
        "current_stage": job_data.get("current_stage", ""),
        "message": job_data.get("message", ""),
        "started_at": job_data.get("started_at"),
        "updated_at": job_data.get("updated_at")
    }
    
    # Add timing information
    if job_data.get("started_at"):
        elapsed = time.time() - job_data["started_at"]
        response["elapsed_seconds"] = int(elapsed)
        
        # Estimate remaining time
        if job_data.get("progress", 0) > 0:
            total_estimated = elapsed / (job_data["progress"] / 100)
            remaining = total_estimated - elapsed
            if remaining > 0:
                response["eta_seconds"] = int(remaining)
    
    # Add completion details
    if job_data.get("status") == "completed":
        response.update({
            "completed_at": job_data.get("completed_at"),
            "results_summary": job_data.get("results_summary", {}),
            "analysis_id": job_data.get("analysis_id")
        })
    elif job_data.get("status") == "failed":
        response.update({
            "failed_at": job_data.get("failed_at"),
            "error": job_data.get("error", "Unknown error")
        })
    elif job_data.get("status") == "cancelled":
        response["cancelled_at"] = job_data.get("cancelled_at")
    
    return response

@router.post("/cancel/{job_id}",
            summary="Cancel Analysis",
            description="Cancel ongoing analysis job")
async def cancel_analysis(
    job_id: str,
    db: Session = Depends(get_db)
):
    """Cancel ongoing analysis job"""
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    
    job_data = analysis_jobs[job_id]
    
    if job_data.get("status") != "processing":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job_data.get('status')}"
        )
    
    # Cancel the asyncio task
    if job_id in active_jobs:
        task = active_jobs[job_id]
        task.cancel()
        del active_jobs[job_id]
    
    # Update job status
    analysis_jobs[job_id].update({
        "status": "cancelled",
        "message": "Analysis cancelled by user",
        "cancelled_at": time.time()
    })
    
    # Update database status
    video_id = job_data.get("video_id")
    if video_id:
        crud.update_video_status(db, video_id, "cancelled")
    
    logger.info(f"Analysis job {job_id} cancelled successfully")
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Analysis cancelled successfully"
    }

@router.get("/jobs",
           summary="List Analysis Jobs",
           description="Get list of all analysis jobs with their current status")
async def list_analysis_jobs():
    """Get list of all analysis jobs"""
    
    jobs = []
    for job_id, job_data in analysis_jobs.items():
        job_summary = {
            "job_id": job_id,
            "video_id": job_data.get("video_id"),
            "status": job_data.get("status"),
            "progress": job_data.get("progress", 0),
            "current_stage": job_data.get("current_stage"),
            "started_at": job_data.get("started_at"),
            "message": job_data.get("message", "")
        }
        
        # Add completion time based on status
        if job_data.get("status") == "completed":
            job_summary["completed_at"] = job_data.get("completed_at")
        elif job_data.get("status") == "failed":
            job_summary["failed_at"] = job_data.get("failed_at")
        elif job_data.get("status") == "cancelled":
            job_summary["cancelled_at"] = job_data.get("cancelled_at")
        
        jobs.append(job_summary)
    
    # Sort by start time (newest first)
    jobs.sort(key=lambda x: x.get("started_at", 0), reverse=True)
    
    return {
        "jobs": jobs,
        "total": len(jobs),
        "active": len([j for j in jobs if j["status"] == "processing"]),
        "completed": len([j for j in jobs if j["status"] == "completed"]),
        "failed": len([j for j in jobs if j["status"] == "failed"])
    }