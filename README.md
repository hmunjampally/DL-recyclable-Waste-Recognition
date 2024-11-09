# Recyclable Waste Image Recognition Based on Deep Learning

## Team Members
- Siri Chandana Garimella
- Hima Sameera Munjampally
- Rohini Kesireddy
- Shagun Sharma

## Problem Statement
With urbanization and increasing population, the amount of domestic waste is rising rapidly, posing serious environmental and logistical challenges. Correctly sorting waste is labor-intensive and prone to errors due to the wide variety of waste types. 

This project aims to develop an intelligent waste classification system using deep learning to automate and improve the accuracy of identifying recyclable waste, helping reduce the environmental impact and enhancing recycling efforts.

## Background
Relevant readings include studies on image classification using deep learning (ResNet), especially for waste sorting. Research by Thung and Yang (2016) on the TrashNet dataset is particularly useful for providing context and data for this project. 

Previous work highlights the limitations of traditional classification methods, which struggle with feature extraction and categorization accuracy, making deep learning an ideal solution.

- [Link to Research](https://www.sciencedirect.com/science/article/abs/pii/S0921344921002457)

## Relevant Reading
- [ResNet50 on Imbalanced Garbage Classification](https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50)
- [VGG19 on Waste Classification](https://ieeexplore.ieee.org/document/9499291/references#references)
- [YOLO5 for Waste Detection](https://www.sciencedirect.com/science/article/abs/pii/S095965262301716X)

## Data Sources
The primary dataset for this project is [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet), which contains images of various waste categories such as cardboard, glass, metal, paper, plastic, and trash. This dataset will be used to train and evaluate the model.

## Methodology
The project will implement a **YOLO (You Only Look Once)** model for real-time object detection in waste images. The YOLO architecture will be enhanced by integrating **LightGlue**, a lightweight feature-matching algorithm. This integration will enable the identification of super points, improving feature matching and recognition, particularly for images captured from challenging angles.

The combined YOLO-LightGlue approach avoids the need for complex manual feature extraction and optimization, instead leveraging automatic feature learning for more accurate object detection and tracking in various waste classification scenarios.

## Evaluation Strategy
1. **Implement YOLO+LightGlue on TrashNet**: Measure mAP, IoU, and F1-score across object classes.
2. **Evaluate feature matching quality**: Use precision-recall curves for keypoints at various view angles.
3. **Conduct ablation study**: Compare YOLO baseline vs. YOLO+LightGlue, focusing on challenging cases.

## Overlapping Projects
- [ResNet50 for Imbalanced Garbage Classification](https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50)
- [VGG19 for Waste Detection](https://ieeexplore.ieee.org/document/9499291/references#references)
- [YOLO5 for Real-Time Waste Classification](https://www.sciencedirect.com/science/article/abs/pii/S095965262301716X)
- 

## Implementation Steps
## Step 1: Setup and Install Dependencies
## 1. Clone the repository:
  
  git clone https://github.com/hmunjampally/DL-recyclable-Waste-Recognition
  
  cd DL-recyclable-Waste-Recognition

## 2. Set up your environment: Ensure you have Python 3.8+ and install necessary dependencies using:
  pip install -r requirements.txt
  
  cd src
  
  git clone https://github.com/ultralytics/yolov5
  
  cd yolov5
  
  pip install -r requirements.txt

## 3. Training YOLOv5
  Navigate to the YOLOv5 Directory: Go to src/yolov5 where YOLOv5 is cloned.
