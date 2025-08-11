from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime, timedelta

from backend.database.session import get_db
from backend.database import models, crud
from backend.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)

# Results caching
results_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 3600  # 1 hour cache

def get_cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """Generate cache key from endpoint and parameters"""
    return f"{endpoint}:{json.dumps(params, sort_keys=True)}"

def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result if still valid"""
    if cache_key in results_cache:
        cached = results_cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL:
            return cached["data"]
        else:
            del results_cache[cache_key]
    return None

def set_cache_result(cache_key: str, data: Dict[str, Any]):
    """Cache result data"""
    results_cache[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }

@router.get("/results/{job_id}",
           summary="Get Analysis Results",
           description="Get comprehensive analysis results with pagination and filtering")
async def get_analysis_results(
    job_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    player_filter: Optional[str] = Query(None, description="Filter by player (jersey number or name)"),
    metric_filter: Optional[str] = Query(None, description="Filter by specific metric"),
    team_filter: Optional[str] = Query(None, description="Filter by team (home/away)"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum score filter"),
    include_details: bool = Query(True, description="Include detailed breakdowns"),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive analysis results with advanced filtering and pagination:
    - Paginated player results and metrics
    - Team-level statistics and comparisons  
    - Timeline of key moments and events
    - Performance benchmarks and recommendations
    - Cached responses for better performance
    """
    
    # Generate cache key
    cache_params = {
        "job_id": job_id,
        "page": page,
        "limit": limit,
        "player_filter": player_filter,
        "metric_filter": metric_filter,
        "team_filter": team_filter,
        "min_score": min_score,
        "include_details": include_details
    }
    cache_key = get_cache_key("results", cache_params)
    
    # Check cache first
    cached_result = get_cached_result(cache_key)
    if cached_result:
        logger.info(f"Returning cached results for {job_id}")
        response = JSONResponse(content=cached_result)
        response.headers["X-Cache-Status"] = "HIT"
        return response
    
    # Find analysis from job_id
    from backend.api.routes.analysis import analysis_jobs
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    
    job_data = analysis_jobs[job_id]
    if job_data.get("status") != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Analysis not completed. Current status: {job_data.get('status')}"
        )
    
    analysis_id = job_data.get("analysis_id")
    video_id = job_data.get("video_id")
    
    if not analysis_id:
        raise HTTPException(status_code=404, detail="Analysis results not found")
    
    # Get analysis
    analysis = crud.get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found in database")
    
    # Get video info
    video = crud.get_video(db, video_id)
    
    # Build filter parameters
    filters = {}
    if team_filter:
        filters["team"] = team_filter
    if min_score is not None:
        filters["min_score"] = min_score
    if player_filter:
        filters["player_filter"] = player_filter
    
    # Get paginated player scores
    player_scores, total_players = crud.get_player_scores_paginated(
        db, 
        analysis_id, 
        page=page, 
        limit=limit,
        filters=filters
    )
    
    # Calculate pagination info
    total_pages = (total_players + limit - 1) // limit
    has_next = page < total_pages
    has_prev = page > 1
    
    # Build response
    result = {
        "job_id": job_id,
        "video_id": video_id,
        "analysis_id": analysis_id,
        "status": "completed",
        "generated_at": datetime.now().isoformat(),
        
        # Pagination info
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total_players,
            "items_per_page": limit,
            "has_next": has_next,
            "has_prev": has_prev
        },
        
        # Summary statistics
        "summary": {
            "total_players_analyzed": total_players,
            "video_duration": video.duration if video else 0,
            "frames_processed": analysis.detections_count,
            "tracks_generated": analysis.tracks_count,
            "actions_classified": analysis.actions_count,
            "analysis_config": job_data.get("config", {})
        },
        
        # Overall scores
        "overall_scores": analysis.scores,
        
        # Paginated player results
        "players": []
    }
    
    # Add player data
    for ps in player_scores:
        player_data = {
            "player_id": ps.player_id,
            "jersey_number": ps.jersey_number,
            "team": ps.team,
            "position": getattr(ps, "position", "Unknown"),
            "scores": {
                "overall": ps.overall_score,
                "biomechanics": ps.biomechanics_score,
                "technical": ps.technical_score,
                "tactical": ps.tactical_score
            }
        }
        
        # Add detailed metrics if requested
        if include_details:
            player_data.update({
                "metrics": {
                    "distance_covered": getattr(ps, "distance_covered", 0),
                    "top_speed": getattr(ps, "top_speed", 0),
                    "average_speed": getattr(ps, "average_speed", 0),
                    "passes_completed": getattr(ps, "passes_completed", 0),
                    "pass_accuracy": getattr(ps, "pass_accuracy", 0),
                    "shots": getattr(ps, "shots", 0),
                    "tackles": getattr(ps, "tackles", 0),
                    "interceptions": getattr(ps, "interceptions", 0)
                },
                "performance": {
                    "strengths": getattr(ps, "strengths", []),
                    "weaknesses": getattr(ps, "weaknesses", []),
                    "recommendations": getattr(ps, "recommendations", [])
                },
                "feedback": ps.feedback
            })
        
        result["players"].append(player_data)
    
    # Add team statistics if on first page
    if page == 1:
        team_stats = crud.get_team_statistics(db, analysis_id)
        result["team_statistics"] = {
            "home_team": team_stats.get("home", {}),
            "away_team": team_stats.get("away", {}),
            "comparison": team_stats.get("comparison", {})
        }
    
    # Add key moments if requested and on first page
    if include_details and page == 1:
        key_moments = crud.get_key_moments(db, analysis_id, limit=10)  # Top 10 moments
        result["key_moments"] = [
            {
                "timestamp": km.timestamp,
                "type": km.event_type,
                "description": km.description,
                "players_involved": km.players_involved,
                "score_impact": km.score_impact,
                "frame_number": getattr(km, "frame_number", 0)
            }
            for km in key_moments
        ]
    
    # Cache the result
    set_cache_result(cache_key, result)
    logger.info(f"Cached results for {job_id}")
    
    response = JSONResponse(content=result)
    response.headers["X-Cache-Status"] = "MISS"
    response.headers["X-Total-Items"] = str(total_players)
    response.headers["X-Page"] = str(page)
    response.headers["X-Total-Pages"] = str(total_pages)
    
    return response

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