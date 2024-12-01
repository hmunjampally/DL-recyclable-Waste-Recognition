import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
            # Look for the latest training results
            results_path = self.config.RESULTS_DIR
            exp_folders = [f for f in os.listdir(results_path) if f.startswith('exp_')]
            
            if not exp_folders:
                print("No trained model found, loading default model...")
                self.model = YOLO('yolov5s.pt')
            else:
                # Get the latest experiment folder
                latest_exp = max(exp_folders, key=lambda x: os.path.getctime(os.path.join(results_path, x)))
                model_path = os.path.join(results_path, latest_exp, 'weights/best.pt')
                
                if os.path.exists(model_path):
                    print(f"Loading trained model from: {model_path}")
                    self.model = YOLO(model_path)
                else:
                    print(f"Trained model not found at {model_path}, loading default model...")
                    self.model = YOLO('yolov5s.pt')
            
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
        
    def test_single_image(self, image_path: str, conf_threshold: float = 0.25) -> None:
        """
        Test the model on a single image and display the results.
        
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold for predictions
        """
        try:
            # Ensure the image exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            print(f"Testing image: {image_path}")
            
            # Perform prediction
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                save=False  # Don't save results to disk
            )
            
            # Get the first result (since we're only processing one image)
            result = results[0]
            
            # Load and display the original image
            img = cv2.imread(image_path)  # type: ignore
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore
            
            plt.figure(figsize=(10, 5))
            
            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot image with predictions
            plt.subplot(1, 2, 2)
            
            # Get prediction boxes, scores, and class indices
            boxes = result.boxes
            
            if len(boxes) > 0:
                # Draw predictions on the image
                img_with_boxes = img.copy()
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Draw rectangle
                    cv2.rectangle(  # type: ignore
                        img_with_boxes,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),  # Green color
                        2
                    )
                    
                    # Add label with confidence
                    label = f"{self.config.CLASS_NAMES[cls]} ({conf:.2f})"
                    cv2.putText(  # type: ignore
                        img_with_boxes,
                        label,
                        (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                plt.imshow(img_with_boxes)
                plt.title('Predictions')
            else:
                plt.imshow(img)
                plt.title('No Detections')
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Print predictions
            print("\nPredictions:")
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    print(f"Detection {i+1}:")
                    print(f"  Class: {self.config.CLASS_NAMES[cls]}")
                    print(f"  Confidence: {conf:.2f}")
            else:
                print("No objects detected.")
                
        except Exception as e:
            print(f"Error during testing: {e}")
            raise