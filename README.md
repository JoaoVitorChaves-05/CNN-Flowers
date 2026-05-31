# 🌸 Flowers Classification with Deep Learning (CNN)

A computer vision project for **flower image classification** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning with MobileNetV2**, achieving **91.47% test accuracy**.

This project was developed as part of a practical Deep Learning assignment and evolved into a complete pipeline including:

✅ Data preprocessing
✅ Transfer Learning + Fine-Tuning
✅ Training and validation pipeline
✅ Classification metrics
✅ Confusion Matrix
✅ ROC/AUC analysis
✅ Error analysis per class
✅ Explainable AI with Grad-CAM

---

## 🚀 Project Overview

The objective of this project is to classify flower images into **5 categories** using a CNN-based architecture optimized for image recognition.

Instead of training a neural network from scratch, **Transfer Learning** was applied using **MobileNetV2 pre-trained on ImageNet**, significantly improving convergence speed and performance.

### 🌼 Flower Classes

* Daisy
* Dandelion
* Roses
* Sunflowers
* Tulips

---

## 🧠 Model Architecture

The project uses **MobileNetV2** as a feature extractor combined with **Fine-Tuning**.

### Why MobileNetV2?

* Lightweight and computationally efficient
* Excellent performance for image classification
* Faster training compared to heavier CNNs
* Suitable for mid-range GPUs

### Training Strategy

* **Transfer Learning**
* **Fine-Tuning** (last layers unfrozen)
* **Data Augmentation**
* **Adam Optimizer**
* **Sparse Categorical Crossentropy**

### Input Shape

```python
(224, 224, 3)
```

---

## 📊 Results

### Final Performance

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **91.47%** |
| Precision | **91.55%** |
| Recall    | **91.47%** |
| F1-Score  | **91.46%** |

### Training Performance

* **20 epochs**
* **~16 minutes training time**
* Strong convergence behavior
* Low overfitting observed

---

## 📈 Generated Visualizations

The pipeline automatically generates several performance visualizations:

### Training Curves

* Accuracy curve
* Loss curve

### Model Evaluation

* Confusion Matrix
* ROC/AUC Curves
* Class Accuracy vs Errors

### Explainable AI

* Grad-CAM heatmaps for interpretability

Generated files:

```txt
output/
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── roc_auc.png
├── class_accuracy_errors.png
└── gradcam/
```

---

## 🔥 Grad-CAM (Explainable AI)

To improve interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was implemented.

This technique highlights **which regions of an image influenced the model prediction**, helping validate whether the CNN actually learned relevant visual patterns.

Example insight:

> The model mainly focused on flower petals and flower centers instead of background regions, indicating meaningful feature learning.

---

## 🏗️ Project Architecture

The codebase follows a **modular software architecture**, improving maintainability and separation of responsibilities.

```txt
project/
│── index.py
│── data_loader.py
│── model.py
│── trainer.py
│── metrics.py
│── visualization.py
│── grad_cam.py
│── model_manager.py
│
├── output/
│
└── saved_model/
```

### Modules Responsibility

| Module             | Responsibility                  |
| ------------------ | ------------------------------- |
| `data_loader.py`   | Dataset loading & preprocessing |
| `model.py`         | CNN architecture                |
| `trainer.py`       | Model training                  |
| `metrics.py`       | Classification metrics          |
| `visualization.py` | Graph generation                |
| `grad_cam.py`      | Explainability with Grad-CAM    |
| `model_manager.py` | Model persistence               |

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-Learn
* KaggleHub

---

## 📦 Dataset

Dataset used:

**Flowers Dataset**

https://www.kaggle.com/datasets/rahmasleam/flowers-dataset

* **3,670 images**
* **5 flower categories**

Train/Test split:

* **70% Training**
* **15% Validation**
* **15% Test**

---

## ⚙️ Running the Project

### Clone repository

```bash
git clone <your-repository-url>
cd CNN-Flowers
```

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run project

```bash
python index.py
```

---

## 💾 Model Persistence

The trained model is automatically saved:

```txt
saved_model/
└── flowers_model.keras
```

This avoids retraining for every execution and speeds up experimentation.

---

## 📌 Key Learnings

This project reinforced practical experience in:

* Computer Vision
* Deep Learning
* CNN architectures
* Transfer Learning
* Fine-Tuning
* Explainable AI (XAI)
* Performance Evaluation
* Modular ML Engineering

---

## 👨‍💻 Author

**João Vitor**

Backend Developer | Software Architecture | Machine Learning Enthusiast

Interested in:

* Backend Engineering
* Deep Learning
* Computer Vision
* AI Systems
* High Performance ML

Feel free to connect on LinkedIn 🚀
