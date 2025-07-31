import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Back, Style

from backend.utils.config import settings

# Initialize colorama for Windows
colorama.init()

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.YELLOW,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        record.name = f"{Fore.BLUE}{record.name}{Style.RESET_ALL}"
        
        # Format message
        message = super().format(record)
        
        return message

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, (level or settings.LOG_LEVEL).upper())
    logger.setLevel(log_level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_uvicorn_logger():
    """Setup uvicorn logger with custom formatting"""
    
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    
    # Clear existing handlers
    uvicorn_logger.handlers.clear()
    uvicorn_access_logger.handlers.clear()
    
    # Setup with our formatter
    setup_logger("uvicorn")
    setup_logger("uvicorn.access")

class AnalysisLogger:
    """Specialized logger for analysis operations"""
    
    def __init__(self, analysis_id: int):
        self.analysis_id = analysis_id
        self.logger = setup_logger(f"analysis.{analysis_id}")
        self.start_time = datetime.now()
    
    def log_stage(self, stage: str, message: str):
        """Log analysis stage"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"[{stage}] {message} (elapsed: {elapsed:.1f}s)")
    
    def log_metrics(self, metrics: dict):
        """Log performance metrics"""
        self.logger.info(f"Metrics: {metrics}")
    
    def log_error(self, stage: str, error: Exception):
        """Log analysis error"""
        self.logger.error(f"[{stage}] Error: {str(error)}", exc_info=True)
    
    def log_completion(self, total_time: float):
        """Log analysis completion"""
        self.logger.info(f"Analysis completed in {total_time:.1f}s")

class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self):
        self.logger = setup_logger("performance")
    
    def log_processing_time(self, operation: str, duration: float, details: dict = None):
        """Log processing time for operations"""
        message = f"{operation}: {duration:.3f}s"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(f"{operation}: {memory_mb:.1f}MB")
    
    def log_gpu_usage(self, operation: str, gpu_memory_mb: float, gpu_utilization: float):
        """Log GPU usage"""
        self.logger.info(
            f"{operation}: GPU Memory: {gpu_memory_mb:.1f}MB, "
            f"Utilization: {gpu_utilization:.1f}%"
        )

def get_analysis_logger(analysis_id: int) -> AnalysisLogger:
    """Get analysis logger instance"""
    return AnalysisLogger(analysis_id)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    return PerformanceLogger()

# Setup root logger
root_logger = setup_logger("football_ai")

# Log system info on startup
def log_system_info():
    """Log system information"""
    import platform
    import psutil
    
    try:
        import torch
        gpu_info = f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "No GPU"
    except ImportError:
        gpu_info = "PyTorch not available"
    
    root_logger.info("="*50)
    root_logger.info("Football AI Analyzer Starting")
    root_logger.info("="*50)
    root_logger.info(f"Platform: {platform.system()} {platform.release()}")
    root_logger.info(f"Python: {platform.python_version()}")
    root_logger.info(f"CPU: {psutil.cpu_count()} cores")
    root_logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    root_logger.info(f"GPU: {gpu_info}")
    root_logger.info(f"Environment: {settings.DEBUG and 'Development' or 'Production'}")
    root_logger.info("="*50)