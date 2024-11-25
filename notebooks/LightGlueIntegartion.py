# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Attempt to import LightGlue
try:
    from lightglue import LightGlue
    from lightglue.utils import load_image, draw_matches
except ModuleNotFoundError:
    print("Error: LightGlue module not found. Please ensure it's installed properly.")
    print("Refer to: https://github.com/cvg/LightGlue for installation instructions.")
    exit(1)

def process_images_with_lightglue(image_dir: Path, output_dir: Path):
    """
    Process a set of images using LightGlue to detect and visualize matches.

    Args:
        image_dir: Path to the directory containing input images.
        output_dir: Path to save output visualizations.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LightGlue
    print("Initializing LightGlue...")
    matcher = LightGlue(pretrained='outdoor')  # Use a pretrained model (indoor/outdoor options)
    
    # Load image pairs for matching
    image_files = sorted(list(image_dir.glob("*.jpg")))
    if len(image_files) < 2:
        print("Error: Not enough images in the directory for matching. At least two required.")
        return
    
    print(f"Found {len(image_files)} images for processing.")
    for i in range(len(image_files) - 1):
        img1_path = image_files[i]
        img2_path = image_files[i + 1]

        print(f"Processing pair: {img1_path.name} and {img2_path.name}")
        img1 = load_image(str(img1_path))
        img2 = load_image(str(img2_path))

        # Match keypoints using LightGlue
        keypoints1, keypoints2, matches = matcher(img1, img2)

        # Visualize matches
        match_viz = draw_matches(img1, img2, keypoints1, keypoints2, matches)

        # Save visualization
        output_path = output_dir / f"matches_{img1_path.stem}_{img2_path.stem}.png"
        plt.imsave(output_path, match_viz)
        print(f"Match visualization saved to: {output_path}")

def main():
    try:
        # Define directories
        input_image_dir = Path("C:/Users/rohin/Desktop/DL-recyclable-Waste-Recognition/dataset/images")
        output_dir = Path("C:/Users/rohin/Desktop/DL-recyclable-Waste-Recognition/output")

        # Verify input directory
        if not input_image_dir.exists() or not any(input_image_dir.glob("*.jpg")):
            print(f"Error: No images found in {input_image_dir}. Ensure the directory contains .jpg files.")
            return

        # Run LightGlue integration
        process_images_with_lightglue(input_image_dir, output_dir)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
