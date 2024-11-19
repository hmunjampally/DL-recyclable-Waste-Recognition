import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

class GarbageClassifier:
    """Garbage classification using YOLOv5."""
    
    def __init__(self, config):
        """Initialize classifier with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get the absolute path to the YOLOv5 directory
        repo_path = Path(__file__).parent.parent / 'yolov5'
        self.yolov5_path = str(repo_path.absolute())
        print(f"yolov5 path {self.yolov5_path}")
        
        self._setup_model()
        self._setup_directories()
    
    def _setup_model(self):
        """Load YOLOv5 model."""
        print("Loading YOLOv5 model...")
        try:
            # Load model using ultralytics
            self.model = YOLO('yolov5su.pt')  # Using the updated model as suggested
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _setup_directories(self):
        """Setup necessary directories."""
        self.results_dir = self.config.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
    
    def train(self, data_yaml: str, epochs: int = None, batch_size: int = None) -> str:
        """Train model and return results path."""
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
        
        epochs = epochs or self.config.DEFAULT_EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE
        
        exp_name = f"exp_{epochs}_epochs"
        exp_path = os.path.join(self.results_dir, exp_name)
        
        print(f"\nStarting training for {epochs} epochs...")
        try:
            # Train the model using ultralytics API
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=self.config.IMAGE_SIZE,
                batch=batch_size,
                project=str(self.results_dir),
                name=exp_name,
                exist_ok=True,
                patience=self.config.PATIENCE
            )
            
            print("Training completed successfully")
            return exp_path
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            return None