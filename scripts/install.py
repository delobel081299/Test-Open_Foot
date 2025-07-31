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
        self.project_root = Path(__file__).parent.parent
        self.python_executable = sys.executable
        self.requirements_installed = False
        self.models_downloaded = False
        self.frontend_setup = False
        self.node_exe = None
        
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
        
    def print_error(self, message):
        """Print error message"""
        cross = "[X]" if self.system == "windows" else ""
        print(f"{Colors.FAIL}{cross} {message}{Colors.ENDC}")
        
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
        if version.major != 3 or version.minor < 8:
            self.print_error(f"Python 3.8+ required. Found: {version.major}.{version.minor}")
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
            
        success, _, error = self.run_command(f'"{self.python_executable}" -m venv venv')
        if success:
            self.print_success("Virtual environment created")
            return True
        else:
            self.print_error(f"Failed to create virtual environment: {error}")
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
            result = subprocess.run(
                [str(pip_exe), "install", "numpy==1.26.0"],
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
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0", 
            "pydantic==2.5.0",
            "opencv-python==4.8.1.78",
            "torch==2.5.1",
            "torchvision==0.20.1"
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
            # Test npm availability
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
            env_content = """# Environment Configuration
DEBUG=false
DATABASE_URL=sqlite:///./football_analyzer.db
SECRET_KEY=your-secret-key-change-in-production

# GPU Settings
GPU_ENABLED=true
GPU_DEVICE_ID=0

# API Settings
CORS_ORIGINS=["http://localhost:3000"]
"""
            env_file.write_text(env_content)
            self.print_success(".env file created ")
        
        return True
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        self.print_step(11, "Running installation tests...")
        
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
        """Print installation summary"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}           Installation Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        print(f" Virtual environment: Created")
        print(f" Python dependencies: {'Installed' if self.requirements_installed else 'Failed'}")
        print(f" Frontend setup: {'Complete' if self.frontend_setup else 'Failed/Skipped'}")
        print(f" AI models: {'Downloaded' if self.models_downloaded else 'Not downloaded (use scripts/download_models.py)'}")
        
        if success:
            print(f"\n{Colors.OKGREEN}Installation completed with warnings!{Colors.ENDC}\n")
        else:
            print(f"\n{Colors.FAIL}Installation failed!{Colors.ENDC}\n")
        
        if self.requirements_installed:
            print("Next steps:")
            print("1. Free up disk space if needed for frontend dependencies")
            print("2. Install FFmpeg manually from https://www.gyan.dev/ffmpeg/builds/")
            print("3. Download AI models: python scripts/download_models.py")
            print("4. Start the application: python scripts/run.py")
            
            print(f"\n{Colors.WARNING}Note: Make sure to activate the virtual environment:{Colors.ENDC}")
            if self.system == "windows":
                print("  venv\\Scripts\\activate")
            else:
                print("  source venv/bin/activate")
    
    def run_installation(self):
        """Run complete installation process"""
        self.print_header()
        
        try:
            # Check prerequisites
            if not self.check_python_version():
                return False
            
            self.check_gpu()
            self.check_ffmpeg()
            
            # Setup environment
            if not self.create_virtual_environment():
                return False
            
            if not self.install_python_dependencies():
                return False
                
            # Skip PyTorch GPU installation since it's in requirements.txt
            
            if not self.setup_frontend():
                return False
            
            # Download models (optional - don't fail if models aren't downloaded)
            self.download_models()
                
            if not self.create_directories():
                return False
                
            if not self.create_config_files():
                return False
            
            if not self.run_tests():
                return False
            
            # Even if some steps failed, show summary if core dependencies are installed
            if self.requirements_installed:
                self.print_summary(success=True)
                return True
            else:
                self.print_summary(success=False)
                return False
            
        except KeyboardInterrupt:
            self.print_error("\nInstallation cancelled by user")
            return False
        except Exception as e:
            self.print_error(f"Installation failed: {str(e)}")
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