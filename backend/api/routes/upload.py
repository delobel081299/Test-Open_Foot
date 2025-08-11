from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import aiofiles
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import asyncio
import time
import os

from backend.database.session import get_db
from backend.database import models, crud
from backend.utils.config import settings
from backend.utils.validators import validate_video_file
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

UPLOAD_DIR = Path("data/uploads")
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

# Upload progress tracking
upload_progress: Dict[str, Dict[str, Any]] = {}

@router.get("/test")
async def test_upload():
    """Test upload endpoint"""
    return {"status": "Upload endpoint is working", "upload_dir": str(UPLOAD_DIR)}

@router.post("/simple")
async def simple_upload(file: UploadFile = File(...)):
    """Simple upload for testing"""
    try:
        logger.info(f"Received file: {file.filename}, size: {file.size}")
        return {
            "success": True, 
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload",
            summary="Upload Video File",
            description="Upload a video file for football analysis with progress tracking")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyze"),
    metadata: Optional[str] = Form(None, description="Additional metadata as JSON string"),
    db: Session = Depends(get_db)
):
    """
    Upload a video file for analysis with enhanced features:
    - Multipart file upload with progress tracking
    - File validation and format checking
    - Duplicate detection via hash comparison
    - Secure file storage with UUID naming
    - Background processing preparation
    """
    
    # Generate job UUID for tracking
    job_id = str(uuid.uuid4())
    
    try:
        # Initialize progress tracking
        upload_progress[job_id] = {
            "status": "uploading",
            "progress": 0,
            "filename": file.filename,
            "started_at": time.time(),
            "error": None
        }
        
        # Check if filename exists
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
        # Validate file extension
        file_extension = Path(str(file.filename)).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON for job {job_id}")
        
        # Generate unique filename
        unique_filename = f"{job_id}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Upload file with progress tracking
        hasher = hashlib.sha256()
        total_size = 0
        
        upload_progress[job_id]["status"] = "processing"
        
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                await f.write(chunk)
                hasher.update(chunk)
                total_size += len(chunk)
                
                # Update progress
                if file.size:
                    progress = min(int((total_size / file.size) * 100), 100)
                    upload_progress[job_id]["progress"] = progress
        
        file_hash = hasher.hexdigest()
        upload_progress[job_id]["progress"] = 100
        upload_progress[job_id]["status"] = "validating"

        # Validate video file (using the UploadFile object, not file_path)
        validation_result = validate_video_file(file)
        if not validation_result.get("valid", False):
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=validation_result.get("error", "Invalid video file"))
        # Check for duplicates
        existing_video = crud.get_video_by_hash(db, file_hash)
        if existing_video:
            file_path.unlink(missing_ok=True)
            upload_progress[job_id]["status"] = "duplicate"
            return JSONResponse(
                status_code=200,
                content={
                    "job_id": job_id,
                    "video_id": existing_video.id,
                    "filename": existing_video.original_filename,
                    "status": "duplicate",
                    "message": "Video already exists in database",
                    "existing_video": {
                        "id": existing_video.id,
                        "upload_date": existing_video.upload_date.isoformat(),
                        "duration": existing_video.duration,
                        "resolution": existing_video.resolution
                    }
                }
            )
        
        # Create database entry
        video = crud.create_video(
            db=db,
            original_filename=str(file.filename),
            saved_filename=unique_filename,
            file_hash=file_hash,
            file_size=total_size,
            duration=validation_result.get("duration", 0),
            fps=validation_result.get("fps", 0),
            resolution=validation_result.get("resolution", ""),
            metadata=parsed_metadata
        )
        
        # Update progress
        upload_progress[job_id].update({
            "status": "completed",
            "progress": 100,
            "video_id": video.id,
            "completed_at": time.time()
        })
        
        logger.info(f"Video uploaded successfully - Job: {job_id}, Video: {video.id}")
        
        return JSONResponse(
            status_code=201,
            content={
                "job_id": job_id,
                "video_id": video.id,
                "filename": video.original_filename,
                "status": "success",
                "message": "Video uploaded successfully",
                "file_info": {
                    "size": total_size,
                    "duration": video.duration,
                    "fps": video.fps,
                    "resolution": video.resolution,
                    "format": file_extension
                },
                "next_step": f"/api/analyze/{job_id}"
            }
        )
        
    except HTTPException:
        # Update progress with error
        upload_progress[job_id]["status"] = "failed"
        raise
    except Exception as e:
        logger.error(f"Upload failed for job {job_id}: {str(e)}")
        # Clean up
        file_path = UPLOAD_DIR / f"{job_id}{file_extension}"
        if file_path.exists():
            file_path.unlink()
        
        upload_progress[job_id].update({
            "status": "failed",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/progress/{job_id}",
           summary="Get Upload Progress",
           description="Get real-time upload progress for a specific job")
async def get_upload_progress(job_id: str):
    """Get upload progress for a specific job ID"""
    
    if job_id not in upload_progress:
        raise HTTPException(status_code=404, detail="Upload job not found")
    
    progress_data = upload_progress[job_id].copy()
    
    # Calculate elapsed time
    if "started_at" in progress_data:
        progress_data["elapsed_time"] = time.time() - progress_data["started_at"]
    
    # Add ETA if uploading
    if progress_data.get("status") == "uploading" and progress_data.get("progress", 0) > 0:
        elapsed = progress_data.get("elapsed_time", 0)
        remaining_progress = 100 - progress_data["progress"]
        if remaining_progress > 0:
            eta = (elapsed / progress_data["progress"]) * remaining_progress
            progress_data["eta_seconds"] = int(eta)
    
    return progress_data

@router.get("/status/{video_id}",
           summary="Get Video Upload Status",
           description="Get detailed status information for an uploaded video")
async def get_upload_status(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get upload status for a video"""
    
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "id": video.id,
        "filename": video.original_filename,
        "upload_date": video.upload_date.isoformat(),
        "file_size": video.file_size,
        "duration": video.duration,
        "fps": video.fps,
        "resolution": video.resolution,
        "status": video.status,
        "metadata": video.metadata if hasattr(video, 'metadata') else {}
    }

@router.delete("/{video_id}")
async def delete_video(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Delete an uploaded video"""
    
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file
    file_path = UPLOAD_DIR / video.saved_filename
    if file_path.exists():
        file_path.unlink()
    
    # Delete database entry
    crud.delete_video(db, video_id)
    
    logger.info(f"Video deleted: {video_id}")
    
    return {"message": "Video deleted successfully"}