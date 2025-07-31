from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.database import models

# Video CRUD operations
def create_video(
    db: Session,
    original_filename: str,
    saved_filename: str,
    file_hash: str,
    file_size: int,
    duration: float,
    fps: float,
    resolution: str
) -> models.Video:
    """Create a new video record"""
    video = models.Video(
        original_filename=original_filename,
        saved_filename=saved_filename,
        file_hash=file_hash,
        file_size=file_size,
        duration=duration,
        fps=fps,
        resolution=resolution
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def get_video(db: Session, video_id: int) -> Optional[models.Video]:
    """Get video by ID"""
    return db.query(models.Video).filter(models.Video.id == video_id).first()

def get_video_by_hash(db: Session, file_hash: str) -> Optional[models.Video]:
    """Get video by file hash"""
    return db.query(models.Video).filter(models.Video.file_hash == file_hash).first()

def get_videos(db: Session, skip: int = 0, limit: int = 100) -> List[models.Video]:
    """Get all videos with pagination"""
    return db.query(models.Video).offset(skip).limit(limit).all()

def update_video_status(db: Session, video_id: int, status: str) -> bool:
    """Update video status"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video:
        video.status = status
        db.commit()
        return True
    return False

def delete_video(db: Session, video_id: int) -> bool:
    """Delete video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video:
        db.delete(video)
        db.commit()
        return True
    return False

# Analysis CRUD operations
def create_analysis(
    db: Session,
    video_id: int,
    scores: Dict[str, Any],
    detections_count: int = 0,
    tracks_count: int = 0,
    actions_count: int = 0,
    processing_time: Optional[float] = None,
    model_versions: Optional[Dict] = None,
    parameters: Optional[Dict] = None
) -> models.Analysis:
    """Create a new analysis record"""
    analysis = models.Analysis(
        video_id=video_id,
        scores=scores,
        detections_count=detections_count,
        tracks_count=tracks_count,
        actions_count=actions_count,
        processing_time=processing_time,
        model_versions=model_versions or {},
        parameters=parameters or {}
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis

def get_analysis(db: Session, analysis_id: int) -> Optional[models.Analysis]:
    """Get analysis by ID"""
    return db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()

def get_analysis_by_video(db: Session, video_id: int) -> Optional[models.Analysis]:
    """Get analysis by video ID"""
    return db.query(models.Analysis).filter(models.Analysis.video_id == video_id).first()

# Player Score CRUD operations
def create_player_score(
    db: Session,
    analysis_id: int,
    player_id: int,
    **kwargs
) -> models.PlayerScore:
    """Create a new player score record"""
    player_score = models.PlayerScore(
        analysis_id=analysis_id,
        player_id=player_id,
        **kwargs
    )
    db.add(player_score)
    db.commit()
    db.refresh(player_score)
    return player_score

def get_player_scores(
    db: Session,
    analysis_id: int,
    team: Optional[str] = None,
    min_score: Optional[float] = None
) -> List[models.PlayerScore]:
    """Get player scores for an analysis"""
    query = db.query(models.PlayerScore).filter(models.PlayerScore.analysis_id == analysis_id)
    
    if team:
        query = query.filter(models.PlayerScore.team == team)
    
    if min_score is not None:
        query = query.filter(models.PlayerScore.overall_score >= min_score)
    
    return query.all()

def get_player_score(db: Session, analysis_id: int, player_id: int) -> Optional[models.PlayerScore]:
    """Get specific player score"""
    return db.query(models.PlayerScore).filter(
        models.PlayerScore.analysis_id == analysis_id,
        models.PlayerScore.player_id == player_id
    ).first()

# Event CRUD operations
def create_event(
    db: Session,
    analysis_id: int,
    timestamp: float,
    frame_number: int,
    event_type: str,
    **kwargs
) -> models.Event:
    """Create a new event record"""
    event = models.Event(
        analysis_id=analysis_id,
        timestamp=timestamp,
        frame_number=frame_number,
        event_type=event_type,
        **kwargs
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event

def get_timeline_events(
    db: Session,
    analysis_id: int,
    event_type: Optional[str] = None
) -> List[models.Event]:
    """Get timeline events for an analysis"""
    query = db.query(models.Event).filter(models.Event.analysis_id == analysis_id)
    
    if event_type:
        query = query.filter(models.Event.event_type == event_type)
    
    return query.order_by(models.Event.timestamp).all()

def get_key_moments(db: Session, analysis_id: int, limit: int = 10) -> List[models.Event]:
    """Get key moments (high impact events)"""
    return db.query(models.Event).filter(
        models.Event.analysis_id == analysis_id
    ).order_by(models.Event.score_impact.desc()).limit(limit).all()

# Report CRUD operations
def create_report(
    db: Session,
    analysis_id: int,
    report_type: str,
    file_path: Optional[str] = None,
    **kwargs
) -> models.Report:
    """Create a new report record"""
    report = models.Report(
        analysis_id=analysis_id,
        report_type=report_type,
        file_path=file_path,
        **kwargs
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

def get_report(db: Session, report_id: int) -> Optional[models.Report]:
    """Get report by ID"""
    return db.query(models.Report).filter(models.Report.id == report_id).first()

def get_reports_by_analysis(db: Session, analysis_id: int) -> List[models.Report]:
    """Get all reports for an analysis"""
    return db.query(models.Report).filter(models.Report.analysis_id == analysis_id).all()

# Team Statistics CRUD operations
def create_team_statistics(
    db: Session,
    analysis_id: int,
    team: str,
    **kwargs
) -> models.TeamStatistic:
    """Create team statistics record"""
    team_stats = models.TeamStatistic(
        analysis_id=analysis_id,
        team=team,
        **kwargs
    )
    db.add(team_stats)
    db.commit()
    db.refresh(team_stats)
    return team_stats

def get_team_statistics(db: Session, analysis_id: int) -> Dict[str, models.TeamStatistic]:
    """Get team statistics for an analysis"""
    stats = db.query(models.TeamStatistic).filter(
        models.TeamStatistic.analysis_id == analysis_id
    ).all()
    
    return {stat.team: stat for stat in stats}

# Heatmap CRUD operations
def create_heatmap_data(
    db: Session,
    analysis_id: int,
    player_id: int,
    positions: List,
    **kwargs
) -> models.HeatmapData:
    """Create heatmap data record"""
    heatmap = models.HeatmapData(
        analysis_id=analysis_id,
        player_id=player_id,
        positions=positions,
        **kwargs
    )
    db.add(heatmap)
    db.commit()
    db.refresh(heatmap)
    return heatmap

def get_player_heatmap(db: Session, analysis_id: int, player_id: int) -> Dict:
    """Get player heatmap data"""
    heatmap = db.query(models.HeatmapData).filter(
        models.HeatmapData.analysis_id == analysis_id,
        models.HeatmapData.player_id == player_id
    ).first()
    
    if heatmap:
        return {
            "positions": heatmap.positions,
            "heatmap_grid": heatmap.heatmap_grid,
            "average_position": heatmap.average_position,
            "area_covered": heatmap.area_covered,
            "time_in_thirds": heatmap.time_in_thirds
        }
    
    return {}

# Analysis utilities
def get_performance_comparisons(db: Session, analysis_id: int) -> Dict:
    """Get performance comparisons for analysis"""
    player_scores = get_player_scores(db, analysis_id)
    team_stats = get_team_statistics(db, analysis_id)
    
    # Calculate comparisons
    home_players = [p for p in player_scores if p.team == "home"]
    away_players = [p for p in player_scores if p.team == "away"]
    
    comparisons = {
        "teams": {
            "home": {
                "avg_score": sum(p.overall_score for p in home_players) / len(home_players) if home_players else 0,
                "player_count": len(home_players)
            },
            "away": {
                "avg_score": sum(p.overall_score for p in away_players) / len(away_players) if away_players else 0,
                "player_count": len(away_players)
            }
        },
        "top_performers": sorted(
            [{"player_id": p.player_id, "score": p.overall_score, "team": p.team} 
             for p in player_scores],
            key=lambda x: x["score"],
            reverse=True
        )[:5],
        "by_position": {},
        "vs_benchmarks": {}
    }
    
    return comparisons

def get_tracks(db: Session, analysis_id: int) -> List:
    """Get tracking data for analysis (placeholder)"""
    # This would be implemented based on how tracking data is stored
    return []