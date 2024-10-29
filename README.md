# **Recyclable Waste Image Recognition Based on Deep Learning**

**Problem Statement:**
      With urbanization and increasing population, the amount of domestic waste is rising rapidly, posing serious environmental and logistical challenges. Correctly sorting waste is labor-intensive and prone to errors due to the wide variety of waste types. This project aims to develop an intelligent waste classification system using deep learning to automate and improve the accuracy of identifying recyclable waste, helping reduce the environmental impact and enhancing recycling efforts.
  
**Background**
  Relevant readings include studies on image classification using deep learning (Reset), especially for waste sorting. Research by Thung and Yang (2016) on the TrashNet dataset is particularly useful for providing context and data for this project. Previous work highlights the limitations of traditional classification methods, which struggle with feature extraction and categorization accuracy, making deep learning an ideal solution.
Link: https://www.sciencedirect.com/science/article/abs/pii/S0921344921002457

**Relevant Reading**
    https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50 – ResNet50
    https://ieeexplore.ieee.org/document/9499291/references#references – VGG19
   https://www.sciencedirect.com/science/article/abs/pii/S095965262301716X – YOLO5

   
**Data Sources**
  The primary dataset for this project is TrashNet (https://www.kaggle.com/datasets/feyzazkefe/trashnet), which contains images of various waste categories such as cardboard, glass, metal, paper, plastic, and trash. This dataset will be used to train and evaluate the model.
  
**Methodology**
  The project will implement a YOLO (You Only Look Once) model for real-time object detection in waste images. The YOLO architecture will be enhanced by integrating LightGlue, a lightweight feature-matching algorithm. This integration will enable the identification of super points, improving feature matching and recognition, particularly for images captured from challenging angles. The combined YOLO-LightGlue approach avoids the need for complex manual feature extraction and optimization, instead leveraging automatic feature learning for more accurate object detection and tracking in various waste classification scenarios.
  
**Evaluation Strategy**
  Implement YOLO+LightGlue on TrashNet, measuring mAP, IoU, and F1-score across object classes.
  Evaluate feature matching quality using precision-recall curves for keypoints at various view angles.
  Conduct ablation study comparing YOLO baseline vs. YOLO+LightGlue, focusing on challenging cases.
  
**Overlapping Projects**
  https://www.kaggle.com/code/farzadnekouei/imbalanced-garbage-classification-resnet50 – ResNet50 was used for this.
  https://ieeexplore.ieee.org/document/9499291/references#references – VGG19 was used for this.
  https://www.sciencedirect.com/science/article/abs/pii/S095965262301716X – YOLO5  was used for this.
