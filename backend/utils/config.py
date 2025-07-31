from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Football AI Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./football_analyzer.db"
    DATABASE_ECHO: bool = False
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # GPU settings
    GPU_ENABLED: bool = True
    GPU_DEVICE_ID: int = 0
    GPU_MEMORY_FRACTION: float = 0.8
    
    # Video settings
    MAX_UPLOAD_SIZE_MB: int = 2000
    SUPPORTED_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv", "webm"]
    OUTPUT_FPS: int = 30
    ANALYSIS_FPS: int = 30
    
    # Model settings
    DETECTION_MODEL: str = "yolov8x"
    POSE_MODEL: str = "mediapipe_heavy"
    ACTION_MODEL: str = "timesformer_base"
    DETECTION_CONFIDENCE: float = 0.5
    TRACKING_CONFIDENCE: float = 0.5
    
    # Analysis settings
    MAX_PLAYERS_TRACKED: int = 22
    MIN_DETECTION_CONFIDENCE: float = 0.6
    ENABLE_3D_POSE: bool = True
    BATCH_SIZE: int = 8
    
    # File paths
    UPLOAD_DIR: str = "data/uploads"
    PROCESSED_DIR: str = "data/processed"
    CACHE_DIR: str = "data/cache"
    REPORTS_DIR: str = "data/reports"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # External APIs (if needed)
    API_RATE_LIMIT: int = 100  # requests per minute
    
    # Performance settings
    MAX_CONCURRENT_ANALYSES: int = 2
    CLEANUP_INTERVAL_HOURS: int = 24
    CACHE_EXPIRY_HOURS: int = 48
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIR,
        settings.PROCESSED_DIR,
        settings.CACHE_DIR,
        settings.REPORTS_DIR,
        settings.MODELS_DIR,
        settings.LOGS_DIR,
        f"{settings.MODELS_DIR}/yolov10",
        f"{settings.MODELS_DIR}/mediapipe",
        f"{settings.MODELS_DIR}/action_recognition",
        f"{settings.MODELS_DIR}/team_classifier"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# GPU configuration
def configure_gpu():
    """Configure GPU settings if available"""
    import torch
    
    if torch.cuda.is_available() and settings.GPU_ENABLED:
        torch.cuda.set_device(settings.GPU_DEVICE_ID)
        # Set memory fraction if using older PyTorch versions
        # torch.cuda.set_per_process_memory_fraction(settings.GPU_MEMORY_FRACTION)
        return True
    return False

# Model paths
def get_model_path(model_type: str, model_name: str) -> str:
    """Get full path to model file"""
    model_mapping = {
        "detection": {
            "yolov8x": f"{settings.MODELS_DIR}/yolov10/yolov8x.pt",
            "yolov10x": f"{settings.MODELS_DIR}/yolov10/yolov10x.pt",
        },
        "pose": {
            "mediapipe_heavy": f"{settings.MODELS_DIR}/mediapipe/pose_landmarker_heavy.task",
            "mediapipe_full": f"{settings.MODELS_DIR}/mediapipe/pose_landmarker_full.task",
        },
        "action": {
            "timesformer_base": f"{settings.MODELS_DIR}/action_recognition/timesformer_base.pth",
        }
    }
    
    return model_mapping.get(model_type, {}).get(model_name, "")

# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    if not Path(settings.UPLOAD_DIR).exists():
        errors.append(f"Upload directory does not exist: {settings.UPLOAD_DIR}")
    
    # Check GPU availability
    if settings.GPU_ENABLED:
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("GPU enabled but CUDA not available")
        except ImportError:
            errors.append("PyTorch not installed but GPU enabled")
    
    # Check model files exist
    detection_model_path = get_model_path("detection", settings.DETECTION_MODEL)
    if detection_model_path and not Path(detection_model_path).exists():
        errors.append(f"Detection model not found: {detection_model_path}")
    
    return errors

# Environment-specific configurations
def get_environment():
    """Determine current environment"""
    return os.getenv("ENVIRONMENT", "development")

def is_production():
    """Check if running in production"""
    return get_environment().lower() == "production"

def is_development():
    """Check if running in development"""
    return get_environment().lower() == "development"

# Initialize on import
ensure_directories()