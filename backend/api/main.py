from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from backend.api.routes import upload, analysis, results, reports
from backend.database.session import engine, Base
from backend.utils.logger import setup_logger
from backend.utils.config import settings

# Setup logging
logger = setup_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Football AI Analyzer API")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Create necessary directories
    directories = [
        Path("data/uploads"),
        Path("data/processed"),
        Path("data/cache"),
        Path("data/reports"),
        Path("models/yolov10"),
        Path("models/mediapipe"),
        Path("models/action_recognition"),
        Path("models/team_classifier"),
        Path("logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Football AI Analyzer API")

# Create FastAPI app
app = FastAPI(
    title="Football AI Analyzer",
    description="Advanced football video analysis using AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])

@app.get("/")
async def root():
    return {
        "message": "Football AI Analyzer API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": settings.GPU_ENABLED,
        "models_loaded": True
    }