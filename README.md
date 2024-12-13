# Standard Echocardiographic View Classification

This repository implements deep learning models for classifying standard echocardiographic views from Transthoracic Echocardiogram (TTE) data. The project explores both image-based and video-based approaches, leveraging state-of-the-art techniques to analyze spatial and temporal dependencies.

---

## **Overview**

### **Image-Based Classification**
This approach uses CNN-based models pre-trained on physiologically significant cardiac phases. The phases were selected using an ECG-assisted multi-phase image sampling technique to ensure the representation of key cardiac activity. The models classify 11 distinct cardiac viewpoints from static TTE images.

- **Key Models:** CNNs pre-trained on ImageNet, fine-tuned on cardiac images.
- **Accuracy:** Achieved 92.4% overall classification accuracy.
- **Sampling Technique:** ECG signals were utilized to identify meaningful cardiac phases for efficient image sampling.

### **Video-Based Classification**
This approach analyzes echocardiographic videos using spatiotemporal information to improve classification performance. EfficientNet-b2 remains as the backbone architecture for extracting spatial features. The following models are used to assess the temporal dependencies between frames.

- **Key Models:**
  - CNN-LSTM
  - (2+1)D Spatiotemporal CNNs
  - Residual-like architectures
  - 3D CNNs
  - Dilated CNNs
  - Multi-Head Self-Attention models

- **Dynamic Weighting:** The contribution of spatial and temporal dependencies to model predictions was dynamically weighted to optimize performance.
- **Micro F1 Score:** Achieved a micro F1 score of 95.1% across 11 cardiac viewpoints.

---

## **Technologies Used**

- **Frameworks:** PyTorch, NumPy, SciPy, Pandas, Transforms, Torchvision
- **Architectures:** CNNs, RNNs, Vision Transformers, 3D CNNs, Attention, Residual connections, Dilated CNNs
- **Preprocessing:** ECG-assisted sampling for image and video classification

---

## **Confusion Matrices**

### Image-Based Classification
![image](https://github.com/user-attachments/assets/3ba5595f-bb1e-40f1-a94e-1ad08773bc60)


### Video-Based Classification
![image](https://github.com/user-attachments/assets/f3649b4b-da8d-42c4-8494-a2fbe30818d3)





---

## **Results**

| Approach          | Metric       | Value     |  Model 
|-------------------|--------------|-----------|-----------
| Image-Based       | Micro F1     | 92.4%     | EfficientNet-b2
| Video-Based       | Micro F1     | 95.1%     | EfficientNet-b2 (Backbone) + Dilated CNNs (Temporal)

---
