from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional, List, Dict

from backend.database.session import get_db
from backend.database import models, crud
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

@router.get("/{video_id}")
async def get_analysis_results(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get complete analysis results for a video"""
    
    # Check if video exists
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get analysis
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Get detailed results
    player_scores = crud.get_player_scores(db, analysis.id)
    team_stats = crud.get_team_statistics(db, analysis.id)
    key_moments = crud.get_key_moments(db, analysis.id)
    
    return {
        "video_id": video_id,
        "analysis_id": analysis.id,
        "overall_scores": analysis.scores,
        "player_scores": [
            {
                "player_id": ps.player_id,
                "jersey_number": ps.jersey_number,
                "team": ps.team,
                "biomechanics_score": ps.biomechanics_score,
                "technical_score": ps.technical_score,
                "tactical_score": ps.tactical_score,
                "overall_score": ps.overall_score,
                "feedback": ps.feedback
            }
            for ps in player_scores
        ],
        "team_statistics": {
            "home_team": team_stats.get("home", {}),
            "away_team": team_stats.get("away", {})
        },
        "key_moments": [
            {
                "timestamp": km.timestamp,
                "type": km.event_type,
                "description": km.description,
                "players_involved": km.players_involved,
                "score_impact": km.score_impact
            }
            for km in key_moments
        ],
        "summary": {
            "total_players_analyzed": len(player_scores),
            "average_score": sum(ps.overall_score for ps in player_scores) / len(player_scores) if player_scores else 0,
            "duration_analyzed": video.duration,
            "frames_processed": analysis.detections_count
        }
    }

@router.get("/{video_id}/players")
async def get_player_results(
    video_id: int,
    team: Optional[str] = None,
    min_score: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get player-specific analysis results"""
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    player_scores = crud.get_player_scores(db, analysis.id, team=team, min_score=min_score)
    
    return {
        "video_id": video_id,
        "players": [
            {
                "player_id": ps.player_id,
                "jersey_number": ps.jersey_number,
                "team": ps.team,
                "position": ps.position,
                "scores": {
                    "biomechanics": ps.biomechanics_score,
                    "technical": ps.technical_score,
                    "tactical": ps.tactical_score,
                    "overall": ps.overall_score
                },
                "metrics": {
                    "distance_covered": ps.distance_covered,
                    "top_speed": ps.top_speed,
                    "passes_completed": ps.passes_completed,
                    "pass_accuracy": ps.pass_accuracy,
                    "shots": ps.shots,
                    "tackles": ps.tackles
                },
                "strengths": ps.strengths,
                "weaknesses": ps.weaknesses,
                "recommendations": ps.recommendations
            }
            for ps in player_scores
        ]
    }

@router.get("/{video_id}/timeline")
async def get_timeline(
    video_id: int,
    event_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get timeline of events from the analysis"""
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    events = crud.get_timeline_events(db, analysis.id, event_type=event_type)
    
    return {
        "video_id": video_id,
        "timeline": [
            {
                "timestamp": event.timestamp,
                "frame_number": event.frame_number,
                "event_type": event.event_type,
                "description": event.description,
                "players_involved": event.players_involved,
                "position": event.position,
                "confidence": event.confidence
            }
            for event in events
        ]
    }

@router.get("/{video_id}/heatmap/{player_id}")
async def get_player_heatmap(
    video_id: int,
    player_id: int,
    db: Session = Depends(get_db)
):
    """Get player position heatmap data"""
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    heatmap_data = crud.get_player_heatmap(db, analysis.id, player_id)
    
    return {
        "video_id": video_id,
        "player_id": player_id,
        "heatmap": heatmap_data.get("positions", []),
        "field_dimensions": {
            "width": 105,
            "height": 68
        },
        "statistics": {
            "average_position": heatmap_data.get("average_position"),
            "area_covered": heatmap_data.get("area_covered"),
            "time_in_thirds": heatmap_data.get("time_in_thirds", {})
        }
    }

@router.get("/{video_id}/comparisons")
async def get_comparisons(
    video_id: int,
    db: Session = Depends(get_db)
):
    """Get performance comparisons between players/teams"""
    
    analysis = crud.get_analysis_by_video(db, video_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    comparisons = crud.get_performance_comparisons(db, analysis.id)
    
    return {
        "video_id": video_id,
        "team_comparison": comparisons.get("teams", {}),
        "top_performers": comparisons.get("top_performers", []),
        "position_comparisons": comparisons.get("by_position", {}),
        "benchmarks": comparisons.get("vs_benchmarks", {})
    }