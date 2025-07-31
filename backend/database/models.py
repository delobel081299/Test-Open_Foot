from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    saved_filename = Column(String, nullable=False, unique=True)
    file_hash = Column(String, nullable=False, unique=True)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    fps = Column(Float, nullable=False)
    resolution = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    
    # Relationships
    analyses = relationship("Analysis", back_populates="video")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis results
    scores = Column(JSON)  # Overall scores
    detections_count = Column(Integer, default=0)
    tracks_count = Column(Integer, default=0)
    actions_count = Column(Integer, default=0)
    
    # Processing metadata
    processing_time = Column(Float)
    model_versions = Column(JSON)
    parameters = Column(JSON)
    
    # Relationships
    video = relationship("Video", back_populates="analyses")
    player_scores = relationship("PlayerScore", back_populates="analysis")
    reports = relationship("Report", back_populates="analysis")
    events = relationship("Event", back_populates="analysis")

class PlayerScore(Base):
    __tablename__ = "player_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # Player identification
    player_id = Column(Integer, nullable=False)  # Track ID
    jersey_number = Column(Integer)
    team = Column(String)  # "home" or "away"
    position = Column(String)
    
    # Scores (0-100)
    biomechanics_score = Column(Float, default=0)
    technical_score = Column(Float, default=0)
    tactical_score = Column(Float, default=0)
    overall_score = Column(Float, default=0)
    
    # Performance metrics
    distance_covered = Column(Float, default=0)  # in meters
    top_speed = Column(Float, default=0)  # in km/h
    passes_completed = Column(Integer, default=0)
    pass_accuracy = Column(Float, default=0)  # percentage
    shots = Column(Integer, default=0)
    tackles = Column(Integer, default=0)
    
    # Analysis details
    strengths = Column(JSON)  # List of strengths
    weaknesses = Column(JSON)  # List of weaknesses
    recommendations = Column(JSON)  # List of recommendations
    feedback = Column(Text)
    
    # Detailed metrics
    biomechanical_data = Column(JSON)
    technical_data = Column(JSON)
    tactical_data = Column(JSON)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="player_scores")

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # Event details
    timestamp = Column(Float, nullable=False)  # seconds from start
    frame_number = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)  # pass, shot, tackle, etc.
    description = Column(Text)
    
    # Players involved
    players_involved = Column(JSON)  # List of player IDs
    primary_player = Column(Integer)  # Main player for this event
    
    # Location
    position = Column(JSON)  # x, y coordinates on field
    field_zone = Column(String)  # attacking_third, middle_third, etc.
    
    # Quality metrics
    confidence = Column(Float, default=0)
    success = Column(Boolean)
    quality_score = Column(Float, default=0)
    score_impact = Column(Float, default=0)  # Impact on overall player score
    
    # Additional data
    metadata = Column(JSON)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="events")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # Report details
    report_type = Column(String, nullable=False)  # pdf, html, video, json
    file_path = Column(String)
    template_id = Column(String)
    language = Column(String, default="en")
    
    # Generation info
    created_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(String)  # user or system
    
    # Report content metadata
    sections = Column(JSON)  # List of included sections
    options = Column(JSON)  # Generation options
    file_size = Column(Integer)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="reports")

class TeamStatistic(Base):
    __tablename__ = "team_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # Team identification
    team = Column(String, nullable=False)  # "home" or "away"
    
    # Team metrics
    possession_percentage = Column(Float, default=0)
    pass_accuracy = Column(Float, default=0)
    shots_total = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    
    # Tactical metrics
    formation = Column(String)
    avg_position = Column(JSON)  # Average team position
    team_width = Column(Float, default=0)  # Team spread
    team_length = Column(Float, default=0)  # Team depth
    
    # Performance scores
    attacking_score = Column(Float, default=0)
    defensive_score = Column(Float, default=0)
    organization_score = Column(Float, default=0)
    
    # Additional statistics
    detailed_stats = Column(JSON)

class HeatmapData(Base):
    __tablename__ = "heatmap_data"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    player_id = Column(Integer, nullable=False)
    
    # Position data
    positions = Column(JSON)  # List of (x, y, timestamp) tuples
    heatmap_grid = Column(JSON)  # 2D grid with intensity values
    
    # Statistics
    average_position = Column(JSON)  # (x, y) average position
    area_covered = Column(Float, default=0)  # in square meters
    time_in_thirds = Column(JSON)  # Time spent in each third
    
    # Field dimensions reference
    field_width = Column(Float, default=105)
    field_height = Column(Float, default=68)