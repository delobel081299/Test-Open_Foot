#!/usr/bin/env python3
"""
Football AI Analyzer - Model Downloader
Downloads required AI models for the application
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
import hashlib
import json
import time

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class ModelDownloader:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        
        # Model definitions with URLs and checksums
        self.models = {
            "yolov8x": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
                "path": "yolov10/yolov8x.pt",
                "size": 136314984,  # bytes
                "description": "YOLOv8 Extra Large model for object detection"
            },
            "yolov8n": {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt", 
                "path": "yolov10/yolov8n.pt",
                "size": 6237136,
                "description": "YOLOv8 Nano model (lightweight)"
            },
            "mediapipe_pose": {
                "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
                "path": "mediapipe/pose_landmarker_heavy.task",
                "size": 12948618,
                "description": "MediaPipe Pose Landmarker (Heavy)"
            }
        }
    
    def print_header(self):
        """Print header"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}    Football AI Analyzer - Model Downloader{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    def create_directories(self):
        """Create model directories"""
        directories = [
            "yolov10",
            "mediapipe", 
            "action_recognition",
            "team_classifier"
        ]
        
        for directory in directories:
            (self.models_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def download_progress_hook(self, block_num, block_size, total_size):
        """Progress hook for urllib.request.urlretrieve"""
        downloaded = block_num * block_size
        
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            downloaded_str = self.format_size(downloaded)
            total_str = self.format_size(total_size)
            
            # Progress bar
            bar_length = 30
            filled_length = int(bar_length * percent / 100)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\r{Colors.OKBLUE}[{bar}] {percent:.1f}% ({downloaded_str}/{total_str}){Colors.ENDC}", end='')
        else:
            downloaded_str = self.format_size(downloaded)
            print(f"\r{Colors.OKBLUE}Downloaded: {downloaded_str}{Colors.ENDC}", end='')
    
    def download_model(self, model_name, model_info):
        """Download a single model"""
        model_path = self.models_dir / model_info["path"]
        
        # Check if model already exists
        if model_path.exists() and model_path.stat().st_size > 0:
            print(f"{Colors.OKGREEN} {model_name} already exists ({self.format_size(model_path.stat().st_size)}){Colors.ENDC}")
            return True
        
        print(f"{Colors.OKBLUE} Downloading {model_name}...{Colors.ENDC}")
        print(f"   {model_info['description']}")
        print(f"   Size: {self.format_size(model_info['size'])}")
        print(f"   URL: {model_info['url']}")
        
        try:
            # Download with progress
            urllib.request.urlretrieve(
                model_info["url"], 
                model_path,
                reporthook=self.download_progress_hook
            )
            print()  # New line after progress bar
            
            # Verify file size
            actual_size = model_path.stat().st_size
            if actual_size != model_info["size"]:
                print(f"{Colors.WARNING} Size mismatch: expected {self.format_size(model_info['size'])}, got {self.format_size(actual_size)}{Colors.ENDC}")
            
            print(f"{Colors.OKGREEN} {model_name} downloaded successfully{Colors.ENDC}")
            return True
            
        except urllib.error.URLError as e:
            print(f"\n{Colors.FAIL} Failed to download {model_name}: {e}{Colors.ENDC}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return False
        except Exception as e:
            print(f"\n{Colors.FAIL} Error downloading {model_name}: {e}{Colors.ENDC}")
            if model_path.exists():
                model_path.unlink()
            return False
    
    def download_all_models(self, models_to_download=None):
        """Download all models or specific models"""
        if models_to_download is None:
            models_to_download = list(self.models.keys())
        
        print(f"{Colors.OKBLUE} Models to download: {', '.join(models_to_download)}{Colors.ENDC}\n")
        
        success_count = 0
        total_count = len(models_to_download)
        
        for model_name in models_to_download:
            if model_name not in self.models:
                print(f"{Colors.WARNING} Unknown model: {model_name}{Colors.ENDC}")
                continue
            
            model_info = self.models[model_name]
            if self.download_model(model_name, model_info):
                success_count += 1
            
            print()  # Spacing between downloads
        
        return success_count, total_count
    
    def list_models(self):
        """List available models"""
        print(f"{Colors.HEADER}Available Models:{Colors.ENDC}\n")
        
        for model_name, model_info in self.models.items():
            model_path = self.models_dir / model_info["path"]
            status = f"{Colors.OKGREEN} Downloaded" if model_path.exists() else f"{Colors.WARNING} Not downloaded"
            
            print(f"{Colors.OKBLUE}{model_name}:{Colors.ENDC}")
            print(f"  Description: {model_info['description']}")
            print(f"  Size: {self.format_size(model_info['size'])}")
            print(f"  Status: {status}{Colors.ENDC}")
            print(f"  Path: {model_info['path']}")
            print()
    
    def check_models(self):
        """Check which models are available"""
        print(f"{Colors.HEADER}Model Status Check:{Colors.ENDC}\n")
        
        downloaded = []
        missing = []
        
        for model_name, model_info in self.models.items():
            model_path = self.models_dir / model_info["path"]
            if model_path.exists() and model_path.stat().st_size > 0:
                downloaded.append(model_name)
                print(f"{Colors.OKGREEN} {model_name} - {self.format_size(model_path.stat().st_size)}{Colors.ENDC}")
            else:
                missing.append(model_name)
                print(f"{Colors.WARNING} {model_name} - Missing{Colors.ENDC}")
        
        print(f"\n{Colors.OKBLUE}Summary:{Colors.ENDC}")
        print(f"  Downloaded: {len(downloaded)}")
        print(f"  Missing: {len(missing)}")
        
        if missing:
            print(f"\n{Colors.WARNING}To download missing models, run:{Colors.ENDC}")
            print(f"  python scripts/download_models.py {' '.join(missing)}")
        
        return len(downloaded), len(missing)
    
    def run(self, args):
        """Main run function"""
        self.print_header()
        self.create_directories()
        
        if not args or args[0] == "all":
            # Download all models
            success, total = self.download_all_models()
        elif args[0] == "list":
            # List available models
            self.list_models()
            return True
        elif args[0] == "check":
            # Check model status
            downloaded, missing = self.check_models()
            return missing == 0
        else:
            # Download specific models
            success, total = self.download_all_models(args)
        
        # Print summary
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}           Download Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        if success == total:
            print(f"{Colors.OKGREEN} All models downloaded successfully! ({success}/{total}){Colors.ENDC}")
            return True
        else:
            print(f"{Colors.WARNING} {success}/{total} models downloaded successfully{Colors.ENDC}")
            if success < total:
                print(f"{Colors.FAIL} {total - success} models failed to download{Colors.ENDC}")
            return False

def main():
    """Main function"""
    downloader = ModelDownloader()
    
    # Parse arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    success = downloader.run(args)
    
    if success:
        print(f"\n{Colors.OKGREEN} Model download completed successfully!{Colors.ENDC}")
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL} Model download failed. Check errors above.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()