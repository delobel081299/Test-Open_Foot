from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict
import io
from pathlib import Path

from backend.database.session import get_db
from backend.database import models, crud
from backend.core.scoring.report_builder import ReportBuilder
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

report_builder = ReportBuilder()

@router.post("/generate/{video_id}")
async def generate_report(
    video_id: int,
    report_type: str = "pdf",
    language: str = "en",
    db: Session = Depends(get_db)
):
    """Generate analysis report"""
    
    # Validate report type
    if report_type not in ["pdf", "html", "json"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid report type. Choose from: pdf, html, json"
        )
    
    # Get video and analysis
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Gather all data for report
    report_data = {
        "video": video,
        "analysis": analysis,
        "player_scores": crud.get_player_scores(db, analysis.id),
        "team_stats": crud.get_team_statistics(db, analysis.id),
        "key_moments": crud.get_key_moments(db, analysis.id),
        "timeline": crud.get_timeline_events(db, analysis.id)
    }
    
    try:
        # Generate report
        if report_type == "pdf":
            report_path = await report_builder.generate_pdf_report(
                report_data,
                language=language
            )
            
            # Save report info to database
            report = crud.create_report(
                db,
                analysis_id=analysis.id,
                report_type="pdf",
                file_path=str(report_path)
            )
            
            return {
                "report_id": report.id,
                "type": "pdf",
                "status": "generated",
                "download_url": f"/api/reports/download/{report.id}"
            }
            
        elif report_type == "html":
            html_content = await report_builder.generate_html_report(
                report_data,
                language=language
            )
            
            return {
                "type": "html",
                "content": html_content
            }
            
        else:  # json
            json_report = await report_builder.generate_json_report(report_data)
            return json_report
            
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

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