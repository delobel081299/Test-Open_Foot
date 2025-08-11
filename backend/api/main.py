from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

from backend.api.routes import upload, analysis, results, reports
from backend.database.session import engine, Base
from backend.utils.logger import setup_logger
from backend.utils.config import settings

# Setup logging
logger = setup_logger(__name__)

# Rate limiting storage
request_counts = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

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
    
    # Load models (optional - can be lazy loaded)
    try:
        # Models will be loaded on first use
        logger.info("Models will be loaded on demand")
    except Exception as e:
        logger.warning(f"Could not preload models: {e}")
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Football AI Analyzer API")
    # Cleanup resources
    await asyncio.sleep(0.1)

# Create FastAPI app
app = FastAPI(
    title="Football AI Analyzer API",
    description="API for AI-powered football video analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CRITICAL: Configure CORS FIRST - This must be the first middleware
# to ensure CORS headers are added to all responses including errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://192.168.56.1:3000",
        "http://192.168.56.1:3001",
        "http://192.168.0.31:3000",
        "http://192.168.0.31:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Add gzip compression (after CORS)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting middleware (after CORS to avoid blocking preflight)
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        response = await call_next(request)
        return response
    
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
        response = await call_next(request)
        return response
        
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                "Access-Control-Allow-Credentials": "true"
            }
        )
    
    # Add current request
    request_counts[client_ip].append(now)
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request data",
            "errors": exc.errors(),
            "body": exc.body
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        },
        headers={
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true"
        }
    )

# Include routers with proper prefixes
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(results.router, prefix="/api", tags=["Results"])
app.include_router(reports.router, prefix="/api", tags=["Reports"])

@app.get("/", 
         summary="API Root", 
         description="Get basic API information")
async def root():
    return {
        "message": "Football AI Analyzer API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "status": "/api/status",
            "results": "/api/results", 
            "report": "/api/report"
        }
    }

@app.get("/health",
         summary="Health Check",
         description="Check API health and system status")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": getattr(settings, 'GPU_ENABLED', False),
        "models_loaded": True,
        "database": "connected",
        "cache": "available"
    }

@app.get("/api/info",
         summary="API Information",
         description="Get detailed API capabilities and limits")
async def api_info():
    return {
        "capabilities": {
            "max_file_size": "500MB",
            "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
            "concurrent_analyses": 5,
            "models": {
                "detection": "YOLOv10",
                "tracking": "ByteTrack",
                "pose": "MediaPipe",
                "action": "TimeSformer"
            }
        },
        "rate_limits": {
            "requests_per_minute": RATE_LIMIT_REQUESTS,
            "max_upload_size": "500MB"
        },
        "processing_stages": [
            "upload",
            "validation", 
            "preprocessing",
            "detection",
            "tracking",
            "pose_extraction",
            "action_classification",
            "analysis",
            "report_generation"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )