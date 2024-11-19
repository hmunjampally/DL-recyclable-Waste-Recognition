import os
import random
import shutil
import platform
from pathlib import Path
import yaml
from typing import Dict

def is_windows() -> bool:
    """Check if the system is Windows."""
    return platform.system() == "Windows"

class DatasetManager:
    """Handles dataset creation and management."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def create_small_dataset(self, original_base_dir: Path, small_base_dir: Path, 
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
        """Create data.yaml for the dataset."""
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
  
        return yaml_path
