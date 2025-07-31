#!/usr/bin/env python3
"""
Football AI Analyzer - Application Runner
Starts both backend and frontend servers
"""

import os
import sys
import subprocess
import threading
import time
import signal
import platform
from pathlib import Path
import webbrowser

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class FootballAIRunner:
    def __init__(self):
        self.system = platform.system().lower()
        self.project_root = Path(__file__).parent.parent
        self.backend_process = None
        self.frontend_process = None
        self.processes = []
        
    def print_header(self):
        """Print startup header"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}    Football AI Analyzer - Starting Application{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    def check_virtual_environment(self):
        """Check if virtual environment exists"""
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            print(f"{Colors.FAIL} Virtual environment not found.{Colors.ENDC}")
            print("Please run the installation script first: python scripts/install.py")
            return False
        return True
    
    def get_python_executable(self):
        """Get Python executable path"""
        if self.system == "windows":
            return self.project_root / "venv" / "Scripts" / "python.exe"
        else:
            return self.project_root / "venv" / "bin" / "python"
    
    def get_pip_executable(self):
        """Get pip executable path"""
        if self.system == "windows":
            return self.project_root / "venv" / "Scripts" / "pip.exe"
        else:
            return self.project_root / "venv" / "bin" / "pip"
    
    def check_dependencies(self):
        """Check if dependencies are installed"""
        print(f"{Colors.OKBLUE} Checking dependencies...{Colors.ENDC}")
        
        python_exe = self.get_python_executable()
        
        # Check if main packages are installed
        try:
            result = subprocess.run([
                str(python_exe), "-c", 
                "import fastapi, uvicorn, torch, cv2; print('Dependencies OK')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{Colors.OKGREEN} Python dependencies found{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.WARNING} Some dependencies missing{Colors.ENDC}")
                return self.install_missing_dependencies()
                
        except Exception as e:
            print(f"{Colors.FAIL} Error checking dependencies: {e}{Colors.ENDC}")
            return False
    
    def install_missing_dependencies(self):
        """Install missing dependencies"""
        print(f"{Colors.WARNING}Installing missing dependencies...{Colors.ENDC}")
        
        pip_exe = self.get_pip_executable()
        requirements_file = self.project_root / "requirements.txt"
        
        try:
            result = subprocess.run([
                str(pip_exe), "install", "-r", str(requirements_file)
            ], check=True)
            
            print(f"{Colors.OKGREEN} Dependencies installed{Colors.ENDC}")
            return True
            
        except subprocess.CalledProcessError:
            print(f"{Colors.FAIL} Failed to install dependencies{Colors.ENDC}")
            return False
    
    def check_models(self):
        """Check if AI models are downloaded"""
        print(f"{Colors.OKBLUE} Checking AI models...{Colors.ENDC}")
        
        models_dir = self.project_root / "models"
        required_models = [
            "yolov10/yolov8x.pt"
        ]
        
        missing_models = []
        for model in required_models:
            model_path = models_dir / model
            if not model_path.exists():
                missing_models.append(model)
        
        if missing_models:
            print(f"{Colors.WARNING} Missing models: {', '.join(missing_models)}{Colors.ENDC}")
            print("Run: python scripts/download_models.py")
            return False
        else:
            print(f"{Colors.OKGREEN} AI models found{Colors.ENDC}")
            return True
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        print(f"{Colors.OKBLUE} Starting backend server...{Colors.ENDC}")
        
        python_exe = self.get_python_executable()
        
        # Change to project root
        os.chdir(self.project_root)
        
        cmd = [
            str(python_exe), "-m", "uvicorn", 
            "backend.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        try:
            self.backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(self.backend_process)
            
            # Monitor backend output in separate thread
            backend_thread = threading.Thread(
                target=self._monitor_process,
                args=(self.backend_process, "BACKEND", Colors.OKGREEN)
            )
            backend_thread.daemon = True
            backend_thread.start()
            
            # Wait a bit for backend to start
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print(f"{Colors.OKGREEN} Backend server started on http://localhost:8000{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL} Backend server failed to start{Colors.ENDC}")
                return False
                
        except Exception as e:
            print(f"{Colors.FAIL} Error starting backend: {e}{Colors.ENDC}")
            return False
    
    def start_frontend(self):
        """Start the React frontend server"""
        print(f"{Colors.OKBLUE} Starting frontend server...{Colors.ENDC}")
        
        frontend_dir = self.project_root / "frontend"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print(f"{Colors.WARNING} Frontend dependencies not found. Installing...{Colors.ENDC}")
            try:
                result = subprocess.run(
                    ["npm", "install"], 
                    cwd=frontend_dir, 
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    if "ENOSPC" in result.stderr:
                        print(f"{Colors.FAIL} Insufficient disk space to install frontend dependencies{Colors.ENDC}")
                        print(f"{Colors.WARNING} Frontend will not be available. Free up disk space and run again.{Colors.ENDC}")
                        print(f"{Colors.WARNING} Backend API is still accessible at http://localhost:8000{Colors.ENDC}")
                        return False
                    else:
                        print(f"{Colors.FAIL} Failed to install frontend dependencies: {result.stderr}{Colors.ENDC}")
                        return False
            except FileNotFoundError:
                print(f"{Colors.FAIL} npm not found. Please install Node.js first.{Colors.ENDC}")
                return False
            except Exception as e:
                print(f"{Colors.FAIL} Error installing frontend dependencies: {e}{Colors.ENDC}")
                return False
        
        try:
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            self.processes.append(self.frontend_process)
            
            # Monitor frontend output in separate thread
            frontend_thread = threading.Thread(
                target=self._monitor_process,
                args=(self.frontend_process, "FRONTEND", Colors.OKCYAN)
            )
            frontend_thread.daemon = True
            frontend_thread.start()
            
            # Wait for frontend to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print(f"{Colors.OKGREEN} Frontend server started on http://localhost:3000{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL} Frontend server failed to start{Colors.ENDC}")
                return False
                
        except Exception as e:
            print(f"{Colors.FAIL} Error starting frontend: {e}{Colors.ENDC}")
            return False
    
    def _monitor_process(self, process, name, color):
        """Monitor process output"""
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"{color}[{name}]{Colors.ENDC} {line.rstrip()}")
    
    def open_browser(self):
        """Open browser to application"""
        print(f"{Colors.OKBLUE} Opening browser...{Colors.ENDC}")
        try:
            webbrowser.open("http://localhost:3000")
        except Exception:
            print(f"{Colors.WARNING} Could not open browser automatically{Colors.ENDC}")
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        print(f"\n{Colors.OKGREEN} Football AI Analyzer is running!{Colors.ENDC}")
        print(f"{Colors.OKBLUE} Frontend: http://localhost:3000{Colors.ENDC}")
        print(f"{Colors.OKBLUE} Backend API: http://localhost:8000{Colors.ENDC}")
        print(f"{Colors.WARNING}Press Ctrl+C to stop the servers{Colors.ENDC}\n")
        
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                if self.backend_process and self.backend_process.poll() is not None:
                    print(f"{Colors.FAIL} Backend process died{Colors.ENDC}")
                    break
                    
                if self.frontend_process and self.frontend_process.poll() is not None:
                    print(f"{Colors.FAIL} Frontend process died{Colors.ENDC}")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING} Shutting down servers...{Colors.ENDC}")
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes"""
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except Exception:
                    pass
        
        print(f"{Colors.OKGREEN} Servers stopped{Colors.ENDC}")
    
    def run(self):
        """Run the complete application"""
        self.print_header()
        
        # Setup signal handler
        def signal_handler(sig, frame):
            print(f"\n{Colors.WARNING} Received shutdown signal{Colors.ENDC}")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Pre-flight checks
            if not self.check_virtual_environment():
                return False
            
            if not self.check_dependencies():
                return False
            
            # Models check is optional - app can run without them
            self.check_models()
            
            # Start servers
            if not self.start_backend():
                return False
            
            # Try to start frontend but continue if it fails
            frontend_started = self.start_frontend()
            
            if frontend_started:
                # Open browser only if frontend started
                self.open_browser()
            else:
                print(f"\n{Colors.WARNING} Frontend not available due to disk space issues.{Colors.ENDC}")
                print(f"{Colors.OKGREEN} Backend API is running at http://localhost:8000{Colors.ENDC}")
                print(f"{Colors.OKBLUE} You can access the API documentation at http://localhost:8000/docs{Colors.ENDC}\n")
            
            # Wait for shutdown
            self.wait_for_shutdown()
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL} Error running application: {e}{Colors.ENDC}")
            self.cleanup()
            return False

def main():
    """Main function"""
    runner = FootballAIRunner()
    
    success = runner.run()
    
    if not success:
        print(f"\n{Colors.FAIL} Failed to start Football AI Analyzer{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()