import os
import random
import shutil

# Paths
#source_dir = r"/Users/sirichandanagarimella/Documents/Fall2024/DeepLearning/DL-recyclable-Waste-Recognition/dataset"
#dest_dir = r"/Users/sirichandanagarimella/Documents/Fall2024/DeepLearning/DL-recyclable-Waste-Recognition/data/TrashNet"
source_dir = "../src/dataset"
dest_dir = "../src/data/trashnet"

# Define classes
class_mapping = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5
}

# Create the target directory structure
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)

# Function to create a YOLO format label
def create_label_file(label_path, class_id):
    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
    # Assuming the object spans most of the image: centered (0.5, 0.5), with full width/height (1.0)
    with open(label_path, 'w') as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Process each class
for category, class_id in class_mapping.items():
    category_dir = os.path.join(source_dir, category)
    images = [img for img in os.listdir(category_dir) if img.endswith(".jpg")]
    random.shuffle(images)
    
    # Calculate split sizes
    train_size = int(len(images) * train_ratio)
    val_size = int(len(images) * val_ratio)
    
    # Assign images to each split
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Copy images and create labels for each split
    for split, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        for image_file in split_images:
            image_source_path = os.path.join(category_dir, image_file)
            image_dest_path = os.path.join(dest_dir, split, 'images', image_file)
            
            # Copy image
            shutil.copy(image_source_path, image_dest_path)
            
            # Create corresponding label file
            label_file = image_file.replace('.jpg', '.txt')
            label_dest_path = os.path.join(dest_dir, split, 'labels', label_file)
            create_label_file(label_dest_path, class_id)

print("Dataset organized and annotations created.")