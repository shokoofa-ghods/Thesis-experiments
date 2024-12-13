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
This approach analyzes echocardiographic videos using spatiotemporal information to improve classification performance.

- **Key Models:**
  - CNN-LSTM
  - (2+1)D Spatiotemporal CNNs
  - Residual-like architectures
  - 3D CNNs
  - Multi-Head Self-Attention models
  - Vision Transformers (ViT)

- **Dynamic Weighting:** The contribution of spatial and temporal dependencies to model predictions was dynamically weighted to optimize performance.
- **Micro F1 Score:** Achieved a micro F1 score of 0.951 across 11 cardiac viewpoints.

---

## **Technologies Used**

- **Frameworks:** PyTorch, NumPy, SciPy, Matplotlib
- **Architectures:** CNNs, RNNs, Vision Transformers, 3D CNNs
- **Preprocessing:** ECG-assisted sampling for image and video classification

---

## **Confusion Matrices**

### Image-Based Classification
![image](https://github.com/user-attachments/assets/3ba5595f-bb1e-40f1-a94e-1ad08773bc60)


### Video-Based Classification
![image](https://github.com/user-attachments/assets/f2f8a12f-be46-4801-b10b-7ef81b96e239)



---

## **How to Run**

### **Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **Dataset Preparation**
1. Organize your dataset into the following structure:
   ```
   dataset/
       images/
           class_1/
           class_2/
           ...
       videos/
           class_1/
           class_2/
           ...
   ```
2. For ECG-assisted sampling, ensure that ECG signal data is available and linked to the corresponding frames/videos.

### **Training**
To train the models:
1. **Image-Based Classification:**
   ```bash
   python train_image_classification.py --dataset_path dataset/images
   ```
2. **Video-Based Classification:**
   ```bash
   python train_video_classification.py --dataset_path dataset/videos
   ```

### **Evaluation**
Run the evaluation scripts to generate the confusion matrices:
```bash
python evaluate_image_classification.py --model_path checkpoints/image_model.pth
python evaluate_video_classification.py --model_path checkpoints/video_model.pth
```

---

## **Results**

| Approach          | Metric       | Value     |  Model 
|-------------------|--------------|-----------|-----------
| Image-Based       | Micro F1     | 92.4%     | EfficientNet-b2
| Video-Based       | Micro F1     | 95.1%     | EfficientNet-b2 (Backbone) + Dilated-CNN (Temporal)

---

## **Future Work**

- **Improving Dataset Size:** Increasing dataset diversity and size to enhance model generalization.
- **Refining ECG Signal Analysis:** Developing robust methods to handle noisy or missing ECG data.
- **Lightweight Models:** Designing lightweight temporal models for real-time applications in clinical settings.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**
Special thanks to the research community and datasets enabling this work.
