import os
import sys
import subprocess
import platform
import torch
from pathlib import Path

def is_windows():
    """Check if the system is Windows."""
    return platform.system() == "Windows"

def run_command(command, check=True, **kwargs):
    """Run a command in a cross-platform manner."""
    if is_windows() and not isinstance(command, str):
        # On Windows, commands need to be joined for shell=True
        command = " ".join(command)

    return subprocess.run(
        command,
        check=check,
        shell=is_windows(),  # Use shell=True on Windows
        **kwargs
    )

def verify_yolov5_installation(yolov5_dir: Path) -> bool:
    """
    Verify if YOLOv5 is properly installed and set up.
    Returns True if installation is valid, False otherwise.
    """
    # Check if directory exists and has essential files
    essential_files = ['train.py', 'detect.py', 'models/yolo.py', 'requirements.txt']
    for file in essential_files:
        if not (yolov5_dir / file).exists():
            return False
    return True

def setup_environment():
    """Setup the environment by checking/installing YOLOv5 and requirements."""
    try:
        # Get the src directory path
        src_dir = Path(__file__).parent.parent.absolute()
        yolov5_dir = src_dir / 'yolov5'

        # Check if YOLOv5 is already properly installed
        if verify_yolov5_installation(yolov5_dir):
            print("YOLOv5 installation verified, skipping reinstallation.")
        else:
            print("Installing YOLOv5...")
            if yolov5_dir.exists():
                print("Found incomplete YOLOv5 installation, reinstalling...")

            # Clone YOLOv5
            run_command([
                'git', 'clone', 'https://github.com/ultralytics/yolov5', str(yolov5_dir)
            ])

        # Verify and install requirements if needed
        requirements_path = str(yolov5_dir / 'requirements.txt')
        if os.path.exists(requirements_path):
            # Read current requirements
            with open(requirements_path, 'r') as f:
                required_packages = [line.strip().split('==')[0].split('>=')[0] 
                                  for line in f.readlines() 
                                  if line.strip() and not line.startswith('#')]

            print("Installing/Updating requirements...")
            if is_windows():
                run_command(f'pip install -r "{requirements_path}"', check=False)
            else:
                run_command(['pip', 'install', '-r', requirements_path], check=False)

        # Ensure ultralytics package is installed
        try:
            import ultralytics
            print(f"Ultralytics version: {ultralytics.__version__}")
        except ImportError:
            print("Installing ultralytics package...")
            run_command(['pip', 'install', 'ultralytics'])

        # Add YOLOv5 to Python path if not already there
        yolov5_path = str(yolov5_dir.absolute())
        if yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)

        # Print environment information
        print("\nEnvironment Setup Complete:")
        print(f"Operating System: {platform.system()}")
        print(f"Python version: {sys.version.split()[0]}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
        print(f"YOLOv5 path: {yolov5_path}")

    except Exception as e:
        print(f"Error setting up environment: {e}")
        raise
