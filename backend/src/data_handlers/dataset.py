import os
import random
import shutil
import platform
from pathlib import Path
import yaml
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_windows() -> bool:
    """Check if the system is Windows."""
    return platform.system() == "Windows"

class DatasetManager:
    """Handles dataset creation and management."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    '''def create_small_dataset(self, original_base_dir: Path, small_base_dir: Path, 
                           samples_per_split: Dict[str, int]) -> Path:
        """Create a smaller version of the dataset for testing."""
        print("Creating small dataset...")

        # Convert paths to Path objects if they're strings
        original_base_dir = Path(original_base_dir)
        small_base_dir = Path(small_base_dir)

        # Create directory structure
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(small_base_dir / split / subdir, exist_ok=True)

        # Copy subset of files for each split
        for split in ['train', 'val', 'test']:
            self._copy_split_files(original_base_dir, small_base_dir,
                                    split, samples_per_split[split])
        
        # Create and save yaml configuration
        yaml_path = self._create_dataset_yaml(small_base_dir)

        print("Small dataset created successfully!")
        return yaml_path

    def _copy_split_files(self, original_base_dir: Path, small_base_dir: Path, 
                         split: str, n_samples: int):
        """Copy files for a specific split."""
        image_dir = original_base_dir / split / 'images'

        # Handle different image extensions
        image_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
        all_images = []
        for pattern in image_patterns:
            all_images.extend(list(image_dir.glob(pattern)))

        if not all_images:
            raise FileNotFoundError(f"No images found in {image_dir}")

        n_samples = min(n_samples, len(all_images))
        selected_images = random.sample(all_images, n_samples)

        print(f"Copying {n_samples} files for {split} split...")

        for img_path in selected_images:
            try:
                # Copy image
                dest_image = small_base_dir / split / 'images' / img_path.name
                shutil.copy2(img_path, dest_image)

                # Copy corresponding label
                label_path = original_base_dir / split / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    dest_label = small_base_dir / split / 'labels' / f"{img_path.stem}.txt"
                    shutil.copy2(label_path, dest_label)
                else:
                    print(f"Warning: No label file found for {img_path.name}")
        
            except Exception as e:
                print(f"Error copying {img_path.name}: {str(e)}")

    def _create_dataset_yaml(self, small_base_dir: Path) -> Path:
        #Create data.yaml for the dataset.
        # Convert paths to absolute paths with forward slashes
        yaml_content = {
            'train': str(small_base_dir.absolute() / 'train' / 'images').replace('\\', '/'),
            'val': str(small_base_dir.absolute() / 'val' / 'images').replace('\\', '/'),
            'test': str(small_base_dir.absolute() / 'test' / 'images').replace('\\', '/'),
            'nc': self.config.NUM_CLASSES,
            'names': self.config.CLASS_NAMES
        }

        yaml_path = small_base_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
  
        return yaml_path'''
    
    def _create_actual_dataset_yaml(self, base_dir: Path) -> Path:
        """Create data.yaml for the actual dataset."""
        self.validate_dataset(base_dir)
        
        # Check if base_dir exists
        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset directory {base_dir} does not exist.")
        
        # Convert paths to absolute paths with forward slashes
        yaml_content = {
            'train': str(base_dir / 'train' / 'images').replace('\\', '/'),
            'val': str(base_dir / 'val' / 'images').replace('\\', '/'),
            'test': str(base_dir / 'test' / 'images').replace('\\', '/'),
            'nc': self.config.NUM_CLASSES,
            'names': self.config.CLASS_NAMES
        }
        # Path to save data.yaml
        yaml_path = base_dir / 'data.yaml'
        
        # Write YAML content to the file
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        logging.info(f"data.yaml created at: {yaml_path}")
        return yaml_path
    
    def validate_dataset(self, base_dir: Path):
        """Ensure all images have corresponding labels and directories exist."""
        for split in ['train', 'val', 'test']:
            images_dir = base_dir / split / 'images'
            labels_dir = base_dir / split / 'labels'
            
            # Ensure directories exist
            if not images_dir.exists() or not labels_dir.exists():
                raise FileNotFoundError(f"Missing required directory: {images_dir} or {labels_dir}")
            
            # Ensure each image has a corresponding label
            for image_path in images_dir.glob("*.jpg"):
                label_path = labels_dir / (image_path.stem + ".txt")
                if not label_path.exists():
                    logging.warning(f"Label missing for image: {image_path.name}")

