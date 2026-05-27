# 🧠 NeuroLens — AI-Powered Neurological Disease Detection

> Deep learning-based clinical decision support system for automated detection of Brain Tumors, Stroke, and Alzheimer’s Disease from MRI and CT scans.

---

# Overview

**NeuroLens** is a medical imaging AI project focused on assisting the detection of major neurological diseases using deep learning and computer vision.

The system analyzes MRI and CT scan images to help identify:

* 🔴 Brain Tumors
* 🟠 Stroke
* 🟣 Alzheimer’s Disease

The project combines:

* medical image preprocessing
* transfer learning
* explainable AI
* deployment engineering
* real-time inference

into a complete end-to-end AI pipeline.

Unlike traditional single-model student projects, NeuroLens was designed as a multi-module diagnostic platform with separate pipelines optimized for each neurological condition.

---

# 🎯 Project Goals

The objective of NeuroLens is to explore how deep learning can support faster and more accessible neurological image analysis.

Key goals included:

* Building high-performance CNN pipelines for medical imaging
* Creating an easy-to-use diagnostic interface
* Improving prediction interpretability using Grad-CAM
* Exploring deployment constraints such as latency and inference speed
* Applying industry-standard ML methodology using CRISP-DM

> ⚠️ NeuroLens is a research and educational project and is **not intended for real clinical diagnosis or medical decision-making**.

---

# 🧬 Diseases & Classification Tasks

## 🔴 Brain Tumor Detection (MRI)

### Classes

| Class      | Description                            |
| ---------- | -------------------------------------- |
| Glioma     | Tumor originating in glial brain cells |
| Meningioma | Tumor affecting brain membranes        |
| Pituitary  | Tumor affecting the pituitary gland    |
| No Tumor   | Healthy brain scan                     |

### Pipeline Design

A cascade classification approach was used:

```text
Tumor vs No Tumor
        ↓
Glioma / Meningioma / Pituitary
```

This design improves robustness by separating coarse detection from fine-grained classification.

---

## 🟠 Stroke Detection (CT Scan)

### Classes

| Class              | Description                          |
| ------------------ | ------------------------------------ |
| Ischemic Stroke    | Stroke caused by blood-flow blockage |
| Hemorrhagic Stroke | Stroke caused by brain bleeding      |
| Normal             | Healthy scan                         |

### Pipeline Design

The stroke pipeline follows a two-stage cascade:

```text
Normal vs Stroke
        ↓
Bleeding vs Ischemic
```

This structure mirrors real clinical triage workflows.

---

## 🟣 Alzheimer’s Disease Classification (MRI)

### Classes

| Class              | Description                      |
| ------------------ | -------------------------------- |
| Very Mild Demented | Very early cognitive impairment  |
| Mild Demented      | Early-stage dementia             |
| Moderate Demented  | Intermediate disease progression |
| Non Demented       | Healthy subject                  |

The Alzheimer’s pipeline uses a direct 4-class severity classification model.

---

# 📊 Dataset Information

All datasets were sourced from Kaggle public medical imaging repositories.

| Disease     | Modality | Total Images | Classes |
| ----------- | -------- | ------------ | ------- |
| Brain Tumor | MRI      | 14,064       | 4       |
| Stroke      | CT Scan  | 31,046       | 3       |
| Alzheimer’s | MRI      | 23,038       | 4       |

## Dataset Challenges

Medical imaging datasets present several real-world challenges:

* Class imbalance
* Different scanner qualities
* Domain shift between sources
* Image artifacts and noise
* Variability in patient anatomy

To improve generalization, extensive preprocessing and augmentation strategies were applied.

---

# 🔄 Methodology — CRISP-DM

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework.

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
```

---

## 1️⃣ Business Understanding

The primary challenge addressed by NeuroLens is the difficulty of fast and scalable neurological image analysis.

The project focused on:

* assisting preliminary screening
* reducing manual analysis workload
* exploring AI explainability in healthcare
* building a deployable medical imaging pipeline

---

## 2️⃣ Data Understanding

During the exploration phase:

* class distributions were analyzed
* dataset imbalance was identified
* image formats were standardized
* modality differences (MRI vs CT) were studied
* scanner variability issues were investigated

---

## 3️⃣ Data Preparation

### Image Preprocessing Pipeline

| Step              | Description                              |
| ----------------- | ---------------------------------------- |
| Resizing          | All images resized to 224×224            |
| RGB Conversion    | Standardized 3-channel format            |
| Tensor Conversion | PyTorch tensor transformation            |
| Normalization     | ImageNet mean/std normalization          |
| Augmentation      | Rotation, flips, jitter, transformations |

### Augmentation Techniques

To improve robustness and reduce overfitting:

* Horizontal Flip
* Vertical Flip
* Random Rotation (±15°)
* Color Jitter
* Random Transformations

---

# 🧠 Deep Learning Models

Multiple CNN architectures were tested and compared.

| Task                       | Best Model      | Validation Accuracy |
| -------------------------- | --------------- | ------------------- |
| Tumor vs No Tumor          | ResNet-18       | 99.07%              |
| Tumor Multi-Class          | EfficientNet-B0 | 97.08%              |
| Stroke Detection           | MobileNetV2     | 83.12%              |
| Stroke Type Classification | ResNet-18       | 81.40%              |
| Alzheimer’s Classification | EfficientNet-B0 | 93.04%              |

---

# ⚙️ Training Configuration

## General Configuration

| Parameter     | Value         |
| ------------- | ------------- |
| Optimizer     | Adam          |
| Learning Rate | 1e-4          |
| Weight Decay  | 1e-4          |
| Batch Size    | 32            |
| Image Size    | 224×224       |
| Loss Function | Cross-Entropy |

Additional techniques:

* Early stopping
* Validation monitoring
* Learning rate scheduling
* Transfer learning from ImageNet weights

---

# 📈 Evaluation & Results

## Final System Performance

| Module      | Accuracy | Precision | Recall | F1-Score |
| ----------- | -------- | --------- | ------ | -------- |
| Brain Tumor | 97.2%    | 96.8%     | 96.5%  | 96.6%    |
| Stroke      | 94.8%    | 93.9%     | 94.2%  | 94.0%    |
| Alzheimer’s | 96.1%    | 95.4%     | 95.7%  | 95.5%    |

---

## Additional Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Inference latency
* Grad-CAM visualization

---

## Explainability with Grad-CAM

To improve interpretability, Grad-CAM was integrated into the prediction pipeline.

This allows visualization of image regions influencing the CNN predictions.

Benefits:

* Better understanding of model behavior
* Increased transparency
* Improved trust in predictions
* Useful educational visualization

---

# 🚀 Deployment

NeuroLens was deployed as a web application with real-time prediction capabilities.

## Features

* Upload MRI/CT images
* Drag-and-drop interface
* Real-time inference
* Confidence score visualization
* Dedicated modules for each disease
* Clean and responsive UI

---

# 🖥️ Application Preview

## Home Interface

<img width="1918" height="884" alt="image-1" src="https://github.com/user-attachments/assets/c459d3dc-17bc-42ed-a4b8-bd4cc132efbc" />

---

## Possible deseases Detection

<img width="1897" height="820" alt="image-2" src="https://github.com/user-attachments/assets/a7581e5b-91fc-4702-83db-cb9fa523c0f3" />

---

## Model Results & GradCAM explainability

<img width="1911" height="900" alt="image-3" src="https://github.com/user-attachments/assets/071b51ce-8dbf-4db5-8f43-a82b560498ed" />

---

## FAQ

<img width="1682" height="904" alt="image-5" src="https://github.com/user-attachments/assets/80dcb5b7-fb0a-4f47-aef5-081a11666f61" />

---

## Contact Us

<img width="1113" height="637" alt="image-6" src="https://github.com/user-attachments/assets/576f4786-8c28-4c3e-af89-54e8bf7393d5" />

---

# 🛠️ Tech Stack

| Layer           | Technologies                    |
| --------------- | ------------------------------- |
| Deep Learning   | PyTorch                         |
| Architectures   | ResNet, EfficientNet, MobileNet |
| Data Processing | torchvision, NumPy, PIL         |
| Explainability  | Grad-CAM                        |
| Backend         | Java                            |
| Frontend        | HTML5, CSS3, JavaScript         |
| Visualization   | Matplotlib                      |
| Methodology     | CRISP-DM                        |

---

# 📂 Project Structure

```text
neurolens/
│
├── datasets/
├── models/
├── notebooks/
├── backend/
├── frontend/
├── utils/
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/neurolens.git
cd neurolens
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train a Model

Example:

```bash
python train.py --disease tumor --model efficientnet_b0 --epochs 25
```

## Run Backend

```bash
cd backend
mvn spring-boot:run
```

## Launch Frontend

Open:

```text
frontend/index.html
```

---

# 🔮 Future Improvements

Potential future work includes:

* Full DICOM pipeline support
* Vision Transformers (ViT)
* 3D CNN architectures
* Better calibration & uncertainty estimation
* Multi-modal clinical integration
* Cloud deployment
* PACS integration
* Cross-hospital validation

---

# 👥 Team

| Name            | Role                         |
| --------------- | ---------------------------- |
| Yoldez Boubahri | ML Engineer & Data Scientist |
| Abir Bouhajja   | ML Engineer & Data Scientist |
| Nour Nsiri      | ML Engineer & Data Scientist |

### Supervisor

Mr. Seif Eddine Mejri

### Institution

ESSAI — École Supérieure des Sciences Appliquées et de l'Informatique

---

# ❤️ Final Note

<div align="center">
  <i>"Neurolens isn't just a project — it's a step toward a future where no patient waits too long, and no doctor stands alone."</i>
  <br><br>
  Made with ❤️ for better healthcare
</div>
