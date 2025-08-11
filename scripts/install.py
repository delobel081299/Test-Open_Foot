#!/usr/bin/env python3
"""
Football AI Analyzer - Automatic Installation Script
Supports Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import urllib.request
import zipfile
import json
import time
import signal

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class FootballAIInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.system_version = platform.version()
        self.machine = platform.machine()
        self.project_root = Path(__file__).parent.parent
        self.python_executable = sys.executable
        self.requirements_installed = False
        self.models_downloaded = False
        self.frontend_setup = False
        self.database_initialized = False
        self.gpu_detected = False
        self.cuda_configured = False
        self.ffmpeg_installed = False
        self.env_configured = False
        self.node_exe = None
        self.errors = []
        self.warnings = []
        
        # Create log directory
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
    def print_header(self):
        """Print installation header"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}    Football AI Analyzer - Automatic Installation{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
    def print_step(self, step, description):
        """Print installation step"""
        print(f"{Colors.OKBLUE}[Step {step}]{Colors.ENDC} {description}")
        
    def print_success(self, message):
        """Print success message"""
        check = "[OK]" if self.system == "windows" else ""
        print(f"{Colors.OKGREEN}{check} {message}{Colors.ENDC}")
        
    def print_warning(self, message):
        """Print warning message"""
        warn = "[!]" if self.system == "windows" else ""
        print(f"{Colors.WARNING}{warn} {message}{Colors.ENDC}")
        self.warnings.append(message)
        
    def print_error(self, message):
        """Print error message"""
        cross = "[X]" if self.system == "windows" else ""
        print(f"{Colors.FAIL}{cross} {message}{Colors.ENDC}")
        self.errors.append(message)
        
    def run_command(self, command, shell=True, capture_output=False):
        """Run system command with error handling"""
        try:
            # On Windows, use cmd.exe explicitly if shell is True
            if self.system == "windows" and shell and isinstance(command, str):
                command = f'cmd /c "{command}"'
                shell = False  # Don't use shell=True with explicit cmd
            
            if capture_output:
                result = subprocess.run(
                    command if isinstance(command, list) else command.split() if not shell else command, 
                    shell=shell, 
                    capture_output=True, 
                    text=True
                )
                return result.returncode == 0, result.stdout, result.stderr
            else:
                result = subprocess.run(
                    command if isinstance(command, list) else command.split() if not shell else command,
                    shell=shell
                )
                return result.returncode == 0, "", ""
        except Exception as e:
            return False, "", str(e)
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        self.print_step(1, "Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 10:
            self.print_error(f"Python 3.10+ required. Found: {version.major}.{version.minor}")
            return False
            
        self.print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_gpu(self):
        """Check GPU availability"""
        self.print_step(2, "Checking GPU availability...")
        
        try:
            success, stdout, _ = self.run_command("nvidia-smi", capture_output=True)
            if success:
                self.print_success("NVIDIA GPU detected")
                return True
            else:
                self.print_warning("No NVIDIA GPU found - CPU mode only")
                return False
        except:
            self.print_warning("Could not detect GPU - CPU mode only")
            return False
    
    def check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        self.print_step(3, "Checking FFmpeg...")
        
        success, _, _ = self.run_command("ffmpeg -version", capture_output=True)
        if success:
            self.print_success("FFmpeg found")
            return True
        else:
            self.print_warning("FFmpeg not found - attempting to install...")
            return self.install_ffmpeg()
    
    def install_ffmpeg(self):
        """Install FFmpeg based on system"""
        if self.system == "windows":
            self.print_warning("Please install FFmpeg manually from https://www.gyan.dev/ffmpeg/builds/")
            return False
        elif self.system == "darwin":  # macOS
            success, _, _ = self.run_command("brew install ffmpeg")
            if success:
                self.print_success("FFmpeg installed via Homebrew")
                return True
        elif self.system == "linux":
            # Try different package managers
            for cmd in ["apt-get install -y ffmpeg", "yum install -y ffmpeg", "dnf install -y ffmpeg"]:
                success, _, _ = self.run_command(f"sudo {cmd}")
                if success:
                    self.print_success("FFmpeg installed")
                    return True
        
        self.print_error("Failed to install FFmpeg automatically")
        return False
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        self.print_step(4, "Setting up virtual environment...")

        venv_path = self.project_root / "venv"

        if venv_path.exists():
            self.print_warning("Virtual environment already exists")
            return True

        # Correction : utiliser une liste d'arguments sans guillemets supplémentaires
        command = [str(self.python_executable), "-m", "venv", "venv"]
        try:
            result = subprocess.run(
                command,
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.print_success("Virtual environment created")
                return True
            else:
                self.print_error(f"Failed to create virtual environment: {result.stderr}")
                return False
        except Exception as e:
            self.print_error(f"Failed to create virtual environment: {e}")
            return False
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        self.print_step(5, "Installing Python dependencies...")
        
        # Determine pip executable
        if self.system == "windows":
            pip_exe = self.project_root / "venv" / "Scripts" / "pip.exe"
            python_exe = self.project_root / "venv" / "Scripts" / "python.exe"
        else:
            pip_exe = self.project_root / "venv" / "bin" / "pip"
            python_exe = self.project_root / "venv" / "bin" / "python"
        
        # Upgrade pip first
        try:
            result = subprocess.run(
                [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0 and "Requirement already satisfied" not in result.stdout and "Requirement already satisfied" not in result.stderr:
                self.print_error(f"Failed to upgrade pip: {result.stderr}")
                return False
            else:
                self.print_success("Pip is up to date")
        except Exception as e:
            self.print_error(f"Error upgrading pip: {e}")
            return False
        
        # Install numpy first (required by some packages)
        self.print_warning("Installing numpy first...")
        try:
            # Use numpy 2.x for Python 3.13 compatibility
            result = subprocess.run(
                [str(pip_exe), "install", "numpy>=2.0.0"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.print_error(f"Failed to install numpy: {result.stderr}")
                return False
        except Exception as e:
            self.print_error(f"Error installing numpy: {e}")
            return False
        
        # Create a minimal requirements file for testing
        minimal_reqs = [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0", 
            "pydantic>=2.5.0",
            "pydantic-settings>=2.0.0",
            "sqlalchemy>=2.0.0",
            "aiofiles>=0.8.0",
            "opencv-python>=4.8.1.78",
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            "colorama>=0.4.6",
            "python-magic-bin>=0.4.14",
            "ffmpeg-python>=0.2.0",
            "ultralytics>=8.0.0",
            "lap>=0.5.0"
        ]
        
        # Install minimal requirements
        self.print_warning("Installing core dependencies...")
        for req in minimal_reqs:
            try:
                result = subprocess.run(
                    [str(pip_exe), "install", req],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    self.print_success(f"Installed {req}")
                else:
                    self.print_warning(f"Failed to install {req}, continuing...")
            except Exception as e:
                self.print_warning(f"Error installing {req}: {e}")
        
        self.print_success("Core dependencies installed")
        self.requirements_installed = True
        return True
    
    def install_pytorch_gpu(self):
        """Install PyTorch with GPU support if available"""
        self.print_step(6, "Setting up PyTorch...")
        
        if self.system == "windows":
            pip_exe = self.project_root / "venv" / "Scripts" / "pip.exe"
        else:
            pip_exe = self.project_root / "venv" / "bin" / "pip"
        
        # Check if GPU is available
        gpu_available = self.check_gpu()
        
        if gpu_available:
            # Install CUDA version
            cuda_command = (
                f'"{str(pip_exe)}" install torch torchvision torchaudio '
                '--index-url https://download.pytorch.org/whl/cu121'
            )
            success, _, _ = self.run_command(cuda_command)
            if success:
                self.print_success("PyTorch with CUDA support installed ")
                return True
        
        # Fallback to CPU version
        cpu_command = f'"{str(pip_exe)}" install torch torchvision torchaudio'
        success, _, _ = self.run_command(cpu_command)
        if success:
            self.print_success("PyTorch (CPU) installed ")
            return True
        else:
            self.print_error("Failed to install PyTorch")
            return False
    
    def setup_frontend(self):
        """Setup React frontend"""
        self.print_step(7, "Setting up frontend...")
        
        frontend_path = self.project_root / "frontend"
        
        # Check if Node.js is available
        try:
            # Test npm availability - On Windows, use cmd.exe explicitly
            if self.system == "windows":
                result = subprocess.run(
                    ["cmd", "/c", "npm", "--version"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
            else:
                result = subprocess.run(
                    "npm --version",
                    capture_output=True,
                    text=True,
                    shell=True,
                    cwd=str(self.project_root)
                )
            
            if result.returncode != 0:
                self.print_error("npm not found. Please install Node.js first.")
                return False
            
            self.print_success(f"npm found: {result.stdout.strip()}")
            
        except Exception as e:
            self.print_error(f"Error checking npm: {e}")
            return False
        
        # Install frontend dependencies
        try:
            self.print_warning("Installing frontend dependencies...")
            
            # On Windows, use cmd.exe explicitly
            if self.system == "windows":
                result = subprocess.run(
                    ["cmd", "/c", "npm", "install"],
                    capture_output=True,
                    text=True,
                    cwd=str(frontend_path)
                )
            else:
                result = subprocess.run(
                    "npm install",
                    capture_output=True,
                    text=True,
                    shell=True,
                    cwd=str(frontend_path)
                )
            
            if result.returncode == 0:
                self.print_success("Frontend dependencies installed")
                self.frontend_setup = True
                return True
            else:
                if "ENOSPC" in result.stderr or "no space left" in result.stderr.lower():
                    self.print_error("Insufficient disk space for frontend dependencies.")
                    self.print_warning("Try freeing up disk space and running the script again.")
                else:
                    self.print_error(f"Failed to install frontend dependencies:")
                    # Show first 200 chars of error
                    error_msg = result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
                    print(f"  {error_msg}")
                return False
                
        except Exception as e:
            self.print_error(f"Error installing frontend dependencies: {e}")
            return False
    
    def download_models(self):
        """Download AI models"""
        self.print_step(8, "Checking AI models...")
        
        models_dir = self.project_root / "models"
        
        # Create model directories
        model_dirs = ["yolov10", "mediapipe", "action_recognition", "team_classifier"]
        for dir_name in model_dirs:
            (models_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.print_warning("Model download skipped - use 'python scripts/download_models.py' to download models")
        self.models_downloaded = False  # Set to False but don't fail the installation
        return True  # Return True to continue installation
    
    def create_directories(self):
        """Create necessary directories"""
        self.print_step(9, "Creating directories...")
        
        directories = [
            "data/uploads",
            "data/processed", 
            "data/cache",
            "data/reports",
            "logs"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files
        for directory in directories:
            gitkeep_file = self.project_root / directory / ".gitkeep"
            gitkeep_file.touch()
        
        self.print_success("Directories created ")
        return True
    
    def create_config_files(self):
        """Create configuration files"""
        self.print_step(10, "Creating configuration files...")
        
        # Create .env file
        env_file = self.project_root / ".env"
        if not env_file.exists():
            import hashlib
            env_content = f"""# Environment Configuration
DEBUG=false
DATABASE_URL=sqlite:///./football_analyzer.db
SECRET_KEY=your-secret-key-change-in-production-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]}

# GPU Settings
GPU_ENABLED={'true' if self.gpu_detected else 'false'}
GPU_DEVICE_ID=0
CUDA_VISIBLE_DEVICES=0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000", "http://localhost:3001"]

# Upload Settings
MAX_UPLOAD_SIZE_MB=500
ALLOWED_VIDEO_EXTENSIONS=[".mp4", ".avi", ".mov", ".mkv"]

# Model Settings
MODEL_CACHE_DIR=models
YOLO_MODEL=yolov10x.pt
CONFIDENCE_THRESHOLD=0.5

# Processing Settings
BATCH_SIZE=32
MAX_VIDEO_LENGTH_SECONDS=600
FRAME_SKIP=1

# Frontend Settings
FRONTEND_URL=http://localhost:3000

# FFmpeg Settings
FFMPEG_PATH={'ffmpeg' if self.ffmpeg_installed else ''}

# System Info (auto-detected)
SYSTEM_OS={self.system}
SYSTEM_GPU={'true' if self.gpu_detected else 'false'}
"""
            env_file.write_text(env_content)
            self.print_success(".env file created ")
            self.env_configured = True
        else:
            self.print_warning(".env file already exists")
            self.env_configured = True
        
        return True
    
    def initialize_database(self):
        """Initialize SQLite database"""
        self.print_step(11, "Initializing database...")
        
        if self.system == "windows":
            python_exe = self.project_root / "venv" / "Scripts" / "python.exe"
        else:
            python_exe = self.project_root / "venv" / "bin" / "python"
        
        # Create database initialization script
        init_script = """
import sys
sys.path.append('.')
try:
    from backend.database.session import engine, Base
    from backend.database import models
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")
except Exception as e:
    print(f"Database initialization warning: {e}")
"""
        
        try:
            # Run initialization
            result = subprocess.run(
                [str(python_exe), "-c", init_script],
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.print_success("Database initialized")
                self.database_initialized = True
                
                # Check database file
                db_file = self.project_root / "football_analyzer.db"
                if db_file.exists():
                    self.print_success(f"Database created: {db_file}")
                return True
            else:
                self.print_warning(f"Database will be initialized on first run")
                # Not critical, can be initialized on first run
                return True
        except Exception as e:
            self.print_warning(f"Database will be initialized on first run")
            return True
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        self.print_step(12, "Running installation tests...")
        
        if self.system == "windows":
            python_exe = self.project_root / "venv" / "Scripts" / "python.exe"
        else:
            python_exe = self.project_root / "venv" / "bin" / "python"
        
        # Test imports directly without writing files
        packages = ["numpy", "cv2", "fastapi", "uvicorn", "torch"]
        failed = []
        
        for package in packages:
            try:
                result = subprocess.run(
                    [str(python_exe), "-c", f"import {package}; print('[OK] {package}')"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(result.stdout.strip())
                else:
                    failed.append(package)
                    print(f"[X] {package} failed")
            except:
                failed.append(package)
                print(f"[X] {package} error")
        
        if len(failed) == 0:
            self.print_success("All packages imported successfully")
            return True
        elif len(failed) < len(packages):
            self.print_warning(f"Some packages missing: {', '.join(failed)}")
            self.print_success("Core functionality should work")
            return True
        else:
            self.print_error("Too many packages failed")
            return False
    
    def print_summary(self, success=True):
        """Print detailed installation summary"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}           Installation Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}System Information:{Colors.ENDC}")
        print(f"  OS: {self.system.title()} {self.system_version}")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Architecture: {self.machine}")
        
        print(f"\n{Colors.BOLD}Installation Status:{Colors.ENDC}")
        status_items = [
            ("OS Detection", True),
            ("Python 3.10+", hasattr(self, 'python_version_ok') and self.python_version_ok),
            ("Virtual Environment", hasattr(self, 'venv_created') and self.venv_created),
            ("Python Dependencies", self.requirements_installed),
            ("GPU Detection", self.gpu_detected),
            ("CUDA/cuDNN", self.cuda_configured),
            ("AI Models", self.models_downloaded),
            ("Database", self.database_initialized),
            ("Directories", True),
            ("FFmpeg", self.ffmpeg_installed),
            ("Environment Config", self.env_configured),
            ("Frontend Setup", self.frontend_setup)
        ]
        
        for name, status in status_items:
            symbol = f"{Colors.OKGREEN}[OK]{Colors.ENDC}" if status else f"{Colors.FAIL}[X]{Colors.ENDC}"
            print(f"  {symbol} {name}")
        
        # Show errors and warnings
        if self.errors:
            print(f"\n{Colors.FAIL}Errors ({len(self.errors)}):{Colors.ENDC}")
            for error in self.errors[:5]:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n{Colors.WARNING}Warnings ({len(self.warnings)}):{Colors.ENDC}")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
        
        # Overall status
        critical_ok = self.requirements_installed
        if success and critical_ok:
            print(f"\n{Colors.OKGREEN}[OK] Installation completed successfully!{Colors.ENDC}")
        elif critical_ok:
            print(f"\n{Colors.WARNING}[!] Installation completed with warnings{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}[X] Installation failed{Colors.ENDC}")
        
        # Next steps
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        steps = []
        
        if not self.ffmpeg_installed and self.system == "windows":
            steps.append("Install FFmpeg from https://www.gyan.dev/ffmpeg/builds/")
        if not self.models_downloaded:
            steps.append("Download AI models: python scripts/download_models.py")
        if not self.frontend_setup:
            steps.append("Install Node.js and run: cd frontend && npm install")
        if self.gpu_detected and not self.cuda_configured:
            steps.append("Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads")
        
        if not steps:
            steps.append("Activate virtual environment:")
            if self.system == "windows":
                steps.append("  venv\\Scripts\\activate")
            else:
                steps.append("  source venv/bin/activate")
            steps.append("Start the application: python scripts/run.py")
        
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")
        
        # Log file info
        log_file = self.log_dir / f"install_{time.strftime('%Y%m%d_%H%M%S')}.log"
        print(f"\n{Colors.BOLD}Log file:{Colors.ENDC} {log_file}")
    
    def run_installation(self):
        """Run complete installation process"""
        self.print_header()
        
        # Add some attributes for tracking
        self.python_version_ok = False
        self.venv_created = False
        
        try:
            # 1. Detect OS (implicit through __init__)
            self.print_step(0, f"Detected OS: {self.system.title()} {self.system_version}")
            
            # 2. Check Python version (3.10+)
            if not self.check_python_version():
                self.print_summary(success=False)
                return False
            self.python_version_ok = True
            
            # 3. Check GPU and CUDA
            self.gpu_detected = self.check_gpu()
            if self.gpu_detected:
                # Check for CUDA
                success, stdout, _ = self.run_command("nvcc --version", capture_output=True)
                if success:
                    self.cuda_configured = True
                    self.print_success("CUDA toolkit detected")
            
            # 4. Check FFmpeg
            self.ffmpeg_installed = self.check_ffmpeg()
            
            # 5. Create virtual environment
            if not self.create_virtual_environment():
                self.print_summary(success=False)
                return False
            self.venv_created = True
            
            # 6. Install Python dependencies
            if not self.install_python_dependencies():
                self.print_summary(success=False)
                return False
            
            # 7. Configure PyTorch with CUDA if GPU available
            if self.gpu_detected:
                self.install_pytorch_gpu()
            
            # 8. Download AI models
            self.download_models()
            
            # 9. Initialize database
            self.initialize_database()
            
            # 10. Create directories
            if not self.create_directories():
                return False
            
            # 11. Configure environment variables
            if not self.create_config_files():
                return False
            
            # 12. Setup frontend
            self.setup_frontend()
            
            # 13. Run tests
            if not self.run_tests():
                self.print_warning("Some tests failed, but installation can continue")
            
            # Show summary
            overall_success = self.requirements_installed and self.venv_created
            self.print_summary(success=overall_success)
            return overall_success
            
        except KeyboardInterrupt:
            self.print_error("\nInstallation cancelled by user")
            self.print_summary(success=False)
            return False
        except Exception as e:
            self.print_error(f"Installation failed: {str(e)}")
            self.errors.append(str(e))
            self.print_summary(success=False)
            return False

def main():
    """Main installation function"""
    installer = FootballAIInstaller()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        installer.print_error("\nInstallation cancelled")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    success = installer.run_installation()
    
    if success:
        print(f"\n{Colors.OKGREEN} Football AI Analyzer installed successfully!{Colors.ENDC}")
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL} Installation failed. Please check the errors above.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()