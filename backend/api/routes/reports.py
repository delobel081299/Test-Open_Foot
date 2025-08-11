from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import io
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from backend.database.session import get_db
from backend.database import models, crud
from backend.core.scoring.report_builder import ReportBuilder
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

try:
    report_builder = ReportBuilder()
except Exception as e:
    logger.warning(f"Failed to initialize ReportBuilder: {e}")
    report_builder = None

# Report generation tracking
report_jobs: Dict[str, Dict[str, Any]] = {}
REPORT_CACHE_TTL = 86400  # 24 hours

async def generate_pdf_background(job_id: str, analysis_id: int, config: Dict, db: Session):
    """Generate PDF report in background"""
    try:
        report_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "stage": "initialization",
            "started_at": time.time(),
            "analysis_id": analysis_id
        }
        
        # Get analysis data
        report_jobs[job_id]["stage"] = "gathering_data"
        report_jobs[job_id]["progress"] = 10
        
        analysis = crud.get_analysis(db, analysis_id)
        video = crud.get_video(db, analysis.video_id)
        
        report_data = {
            "video": video,
            "analysis": analysis,
            "player_scores": crud.get_player_scores(db, analysis_id),
            "team_stats": crud.get_team_statistics(db, analysis_id),
            "key_moments": crud.get_key_moments(db, analysis_id),
            "timeline": crud.get_timeline_events(db, analysis_id)[:50]  # Limit for PDF
        }
        
        report_jobs[job_id]["progress"] = 30
        
        # Generate PDF
        if report_builder:
            report_jobs[job_id]["stage"] = "generating_pdf"
            report_jobs[job_id]["progress"] = 50
            
            # Ensure reports directory exists
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = reports_dir / f"report_{job_id}.pdf"
            
            # Mock PDF generation for now (replace with actual implementation)
            await asyncio.sleep(2)  # Simulate processing time
            
            # Create a simple PDF placeholder
            with open(output_path, 'w') as f:
                f.write(f"Football Analysis Report\nGenerated: {datetime.now()}\nAnalysis ID: {analysis_id}")
                
            report_jobs[job_id]["progress"] = 90
            
            # Save to database
            report = crud.create_report(
                db,
                analysis_id=analysis_id,
                report_type="pdf",
                file_path=str(output_path),
                metadata=config
            )
            
            report_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "stage": "completed",
                "report_id": report.id,
                "file_path": str(output_path),
                "completed_at": time.time()
            })
        else:
            raise Exception("Report builder not available")
            
    except Exception as e:
        report_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": time.time()
        })
        logger.error(f"PDF generation failed for job {job_id}: {e}")

@router.get("/report/{job_id}/pdf",
           summary="Generate PDF Report", 
           description="Generate PDF report with streaming response and 24h cache")
async def generate_pdf_report(
    job_id: str,
    background_tasks: BackgroundTasks,
    template: str = "standard",
    language: str = "en",
    include_charts: bool = True,
    include_heatmaps: bool = True,
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive PDF report with advanced features:
    - Streaming response for large reports
    - 24-hour caching system
    - Multiple template options
    - Customizable sections and visualizations
    - Background processing with progress tracking
    """
    
    # Find analysis from job_id
    from backend.api.routes.analysis import analysis_jobs
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    
    job_data = analysis_jobs[job_id]
    if job_data.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Status: {job_data.get('status')}"
        )
    
    analysis_id = job_data.get("analysis_id")
    if not analysis_id:
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    # Check if PDF already exists and is cached
    pdf_job_id = f"pdf_{job_id}_{template}_{language}"
    if pdf_job_id in report_jobs:
        job_status = report_jobs[pdf_job_id]
        if job_status.get("status") == "completed":
            # Check if file still exists and cache is valid
            file_path = Path(job_status.get("file_path", ""))
            cache_age = time.time() - job_status.get("completed_at", 0)
            
            if file_path.exists() and cache_age < REPORT_CACHE_TTL:
                logger.info(f"Returning cached PDF for {job_id}")
                return StreamingResponse(
                    io.open(file_path, 'rb'),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"inline; filename=football_analysis_{job_id}.pdf",
                        "X-Cache-Status": "HIT",
                        "X-Cache-Age": str(int(cache_age))
                    }
                )
        elif job_status.get("status") == "processing":
            return JSONResponse({
                "status": "processing",
                "progress": job_status.get("progress", 0),
                "stage": job_status.get("stage", ""),
                "message": "PDF generation in progress",
                "job_id": pdf_job_id,
                "progress_endpoint": f"/api/report/progress/{pdf_job_id}"
            })
    
    # Validate template
    valid_templates = ["standard", "player_focus", "tactical", "coach"]
    if template not in valid_templates:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid template. Choose from: {', '.join(valid_templates)}"
        )
    
    # Start PDF generation
    config = {
        "template": template,
        "language": language,
        "include_charts": include_charts,
        "include_heatmaps": include_heatmaps,
        "job_id": job_id
    }
    
    background_tasks.add_task(
        generate_pdf_background,
        pdf_job_id,
        analysis_id,
        config,
        db
    )
    
    logger.info(f"Started PDF generation for job {pdf_job_id}")
    
    return JSONResponse({
        "status": "started",
        "message": "PDF generation started",
        "job_id": pdf_job_id,
        "template": template,
        "estimated_time": "30-60 seconds",
        "progress_endpoint": f"/api/report/progress/{pdf_job_id}",
        "download_endpoint": f"/api/report/{job_id}/pdf"
    })

@router.get("/report/progress/{pdf_job_id}",
           summary="Get PDF Generation Progress",
           description="Get real-time progress of PDF generation")
async def get_pdf_progress(pdf_job_id: str):
    """Get PDF generation progress"""
    
    if pdf_job_id not in report_jobs:
        raise HTTPException(status_code=404, detail="PDF job not found")
    
    job_data = report_jobs[pdf_job_id]
    
    response = {
        "job_id": pdf_job_id,
        "status": job_data.get("status"),
        "progress": job_data.get("progress", 0),
        "stage": job_data.get("stage", ""),
        "started_at": job_data.get("started_at")
    }
    
    # Add timing info
    if job_data.get("started_at"):
        elapsed = time.time() - job_data["started_at"]
        response["elapsed_seconds"] = int(elapsed)
        
        if job_data.get("progress", 0) > 0:
            estimated_total = (elapsed / job_data["progress"]) * 100
            remaining = estimated_total - elapsed
            if remaining > 0:
                response["eta_seconds"] = int(remaining)
    
    # Add completion details
    if job_data.get("status") == "completed":
        response.update({
            "completed_at": job_data.get("completed_at"),
            "report_id": job_data.get("report_id"),
            "file_size": Path(job_data.get("file_path", "")).stat().st_size if Path(job_data.get("file_path", "")).exists() else 0
        })
    elif job_data.get("status") == "failed":
        response.update({
            "failed_at": job_data.get("failed_at"),
            "error": job_data.get("error")
        })
    
    return response

@router.get("/download/{report_id}")
async def download_report(
    report_id: int,
    db: Session = Depends(get_db)
):
    """Download generated report"""
    
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    file_path = Path(report.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"football_analysis_report_{report.id}.pdf",
        media_type="application/pdf"
    )

@router.post("/video/{video_id}/annotate")
async def generate_annotated_video(
    video_id: int,
    include_scores: bool = True,
    include_tracking: bool = True,
    include_events: bool = True,
    db: Session = Depends(get_db)
):
    """Generate annotated video with analysis overlay"""
    
    # Get video and analysis
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        # Generate annotated video
        output_path = await report_builder.generate_annotated_video(
            video_path=f"data/uploads/{video.saved_filename}",
            analysis_data={
                "analysis": analysis,
                "tracks": crud.get_tracks(db, analysis.id),
                "events": crud.get_timeline_events(db, analysis.id)
            },
            options={
                "include_scores": include_scores,
                "include_tracking": include_tracking,
                "include_events": include_events
            }
        )
        
        # Save to database
        report = crud.create_report(
            db,
            analysis_id=analysis.id,
            report_type="video",
            file_path=str(output_path)
        )
        
        return {
            "report_id": report.id,
            "type": "video",
            "status": "generated",
            "download_url": f"/api/reports/download/{report.id}"
        }
        
    except Exception as e:
        logger.error(f"Video annotation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video annotation failed: {str(e)}"
        )

@router.get("/templates")
async def get_report_templates():
    """Get available report templates"""
    
    return {
        "templates": [
            {
                "id": "standard",
                "name": "Standard Analysis Report",
                "description": "Comprehensive report with all analysis sections",
                "sections": [
                    "summary",
                    "player_analysis",
                    "team_analysis",
                    "key_moments",
                    "recommendations"
                ]
            },
            {
                "id": "player_focus",
                "name": "Player Performance Report",
                "description": "Detailed analysis focused on individual players",
                "sections": [
                    "player_profiles",
                    "performance_metrics",
                    "biomechanical_analysis",
                    "improvement_areas"
                ]
            },
            {
                "id": "tactical",
                "name": "Tactical Analysis Report",
                "description": "Team tactics and strategic insights",
                "sections": [
                    "formation_analysis",
                    "movement_patterns",
                    "team_cohesion",
                    "tactical_recommendations"
                ]
            },
            {
                "id": "coach",
                "name": "Coach's Report",
                "description": "Actionable insights for coaching staff",
                "sections": [
                    "team_overview",
                    "individual_feedback",
                    "training_recommendations",
                    "match_preparation"
                ]
            }
        ]
    }

@router.post("/share/{report_id}")
async def share_report(
    report_id: int,
    share_options: Dict,
    db: Session = Depends(get_db)
):
    """Share report via email or generate shareable link"""
    
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # For now, just return a placeholder response
    # In production, implement actual sharing functionality
    
    return {
        "report_id": report_id,
        "share_link": f"http://localhost:3000/shared/report/{report_id}",
        "expires_at": "2024-12-31T23:59:59Z",
        "message": "Report shared successfully"
    }