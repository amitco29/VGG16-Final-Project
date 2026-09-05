# ✈️ Aircraft Image Classification (VGG16)

This repository contains an end-to-end pipeline for an 8-class aircraft image classification model. Built on the VGG16 architecture using Transfer Learning, the project leverages a custom, perfectly balanced dataset of 960 RGB images. It features advanced data augmentation techniques, hyperparameter optimization, and deep model analysis utilizing Unsupervised Manifold Learning (UMAP) for feature space visualization and Grad-CAM for model explainability.

## 🚀 Key Features & Methodologies
* **Architecture:** VGG16 with Transfer Learning.
* **Data Augmentation:** Implemented **CutOut** to improve model robustness and prevent overfitting.
* **Optimization & Validation:** Extensive hyperparameter tuning using **Grid Search** and robust evaluation via **5-fold Cross-Validation**.
* **Explainable AI (XAI):** 
  * **Grad-CAM:** Visualized the network's focus areas to ensure the model learns relevant aircraft features rather than background noise.
  * **UMAP:** Projected high-dimensional feature spaces into 2D to validate class separation.

## 📊 Dataset
* **Classes:** 8 distinct aircraft categories.
* **Size:** 960 RGB images.
* **Distribution:** Perfectly balanced across all classes.

## 💻 Getting Started

### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn umap-learn opencv-python
