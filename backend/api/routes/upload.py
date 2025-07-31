from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import aiofiles
import hashlib
from pathlib import Path
from typing import Optional
import uuid

from backend.database.session import get_db
from backend.database import models, crud
from backend.utils.config import settings
from backend.utils.validators import validate_video_file
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

UPLOAD_DIR = Path("data/uploads")

@router.post("/")
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a video file for analysis"""
    
    # Validate file
    validation_result = validate_video_file(file)
    if not validation_result["valid"]:
        raise HTTPException(status_code=400, detail=validation_result["error"])
    
    # Generate unique filename
    file_extension = Path(file.filename).suffix
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}{file_extension}"
    file_path = UPLOAD_DIR / filename
    
    # Calculate file hash
    hasher = hashlib.sha256()
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
                hasher.update(chunk)
        
        file_hash = hasher.hexdigest()
        
        # Check if file already exists
        existing_video = crud.get_video_by_hash(db, file_hash)
        if existing_video:
            # Remove duplicate file
            file_path.unlink()
            return {
                "id": existing_video.id,
                "filename": existing_video.original_filename,
                "status": "duplicate",
                "message": "Video already uploaded"
            }
        
        # Create database entry
        video = crud.create_video(
            db,
            original_filename=file.filename,
            saved_filename=filename,
            file_hash=file_hash,
            file_size=file_path.stat().st_size,
            duration=validation_result.get("duration", 0),
            fps=validation_result.get("fps", 0),
            resolution=validation_result.get("resolution", "")
        )
        
        logger.info(f"Video uploaded successfully: {video.id}")
        
        return {
            "id": video.id,
            "filename": video.original_filename,
            "status": "success",
            "message": "Video uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        # Clean up file if upload failed
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/status/{video_id}")
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
        "upload_date": video.upload_date,
        "file_size": video.file_size,
        "duration": video.duration,
        "fps": video.fps,
        "resolution": video.resolution,
        "status": video.status
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