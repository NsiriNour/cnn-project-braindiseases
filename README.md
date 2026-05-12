# 🧠 NeuroLens — AI-Powered Neurological Disease Detection

> **Automated Detection of Brain Tumors, Stroke & Alzheimer's Disease using Deep Learning**


## Overview

**NeuroLens** is an AI-powered clinical decision support system designed to automate the detection of three critical neurological conditions from medical imaging scans (MRI and CT). Built following the 

**Core Goals:**
- Enhance clinical decision-making with **>95% precision**
- Reduce time-to-treatment by **80%**
- Bridge the expert gap by bringing advanced diagnostics to every healthcare facility

---

## Diseases & Classes

### 🔴 Brain Tumor (MRI)
| Class | Type | Description |
|-------|------|-------------|
| Glioma | Malignant | Tumor originating in brain glial cells |
| Meningioma | Membrane | Tumor developing in brain membranes |
| Pituitary | Gland | Tumor affecting the pituitary gland |
| No Tumor | — | Healthy brain scan |

### 🟠 Stroke (CT Scan)
| Class | Type | Description |
|-------|------|-------------|
| Ischemic Stroke | Blockage | Stroke caused by blood flow blockage due to a clot or narrowed artery |
| Hemorrhagic Stroke | Bleeding | Stroke caused by rupture of a blood vessel leading to brain bleeding |
| Normal | — | Healthy brain scan |

### 🟣 Alzheimer's Disease (MRI)
| Class | Stage | Description |
|-------|-------|-------------|
| Very Mild Impairment | Early | Very early signs of dementia |
| Mild Impairment | Stage 1 | Early stage cognitive impairment |
| Moderate Impairment | Stage 2 | Intermediate disease progression |
| No Impairment | — | No cognitive decline detected |

---

## Dataset

All datasets sourced from **Kaggle**:

| Disease | Modality | Total Images | Classes | Train Split | Test Split |
|---------|----------|-------------|---------|-------------|------------|
| Brain Tumor | MRI | 14,064 | 4 | 11,424 | 2,622 |
| Stroke | CT Scan | 31,046 | 3 | ~28,000 | ~3,000 |
| Alzheimer's | MRI | 23,038 | 4 | 20,480 | 2,558 |

**Key data observations:**
- Class imbalance across diagnostic categories → addressed via augmentation
- Scanner/site variability (domain shift) → addressed via normalization

---

## Methodology — CRISP-DM

This project strictly follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** — a 6-phase iterative framework:

```
Business Understanding → Data Understanding → Data Preparation
         ↑                                              ↓
      Deployment  ←  Evaluation  ←  Modeling  ←────────┘
```

### Phase 1 — Business Understanding
Defined the clinical problem, target diseases, and success metrics (>95% accuracy, <0.5s inference).

### Phase 2 — Data Understanding
Explored class distributions, file types (JPG, PNG, DICOM), train/test splits, and identified domain shift issues.

### Phase 3 — Data Preparation
| Step | Technique |
|------|-----------|
| Image Resizing | All images resized to `224×224` pixels |
| RGB Conversion | Uniform 3-channel color format |
| Tensor Transformation | Converted to tensors for CNN pipeline |
| Pixel Normalization | ImageNet mean & std normalization |
| Augmentation | Random flip (H/V), rotation (±15°), color jitter |

### Phase 4 — Modeling
Three cascade CNN pipelines were developed:

- **Brain Tumor CNN** — Cascade: Binary (Tumor/No Tumor) → 3-class (Glioma/Meningioma/Pituitary)
- **Stroke CNN** — Cascade: Binary (Normal/Stroke) → Binary (Bleeding/Ischemia)
- **Alzheimer's CNN** — Single model: 4-class severity grading

### Phase 5 — Evaluation
Models evaluated on accuracy, precision, recall, F1-score, AUC, Grad-CAM explainability, and inference latency.

### Phase 6 — Deployment
Deployed as a web application (**NeuroLens**) with a clean interface for clinical use.

---

## Model Architecture

| Disease | Task | Best Model | Val Accuracy | Parameters | Inference |
|---------|------|-----------|-------------|------------|-----------|
| Brain Tumor (Binary) | Tumor vs No Tumor | ResNet-18 | **99.07%** | 11.2M | 0.31s |
| Brain Tumor (Multi) | 3-class | EfficientNet-B0 | **97.08%** | 4.2M | 0.31s |
| Stroke (Binary) | Normal vs Stroke | MobileNetV2 | **83.12%** | 2.2M | 0.23s |
| Stroke (Binary) | Bleeding vs Ischemia | ResNet-18 | **81.40%** | 11.2M | 0.23s |
| Alzheimer's | 4-class staging | EfficientNet-B0 | **93.04%** | 4.2M | 0.44s |

**Training configuration (all models):**
- Optimizer: Adam (`lr=1e-4`, `wd=1e-4`)
- Loss: Cross-Entropy (+ Macro F1 for Alzheimer's)
- Batch size: 32 | Image size: 224×224
- Early stopping with patience

---

## Results

### Summary Table
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Brain Tumor | **97.2%** | 96.8% | 96.5% | 96.6% |
| Stroke | **94.8%** | 93.9% | 94.2% | 94.0% |
| Alzheimer's | **96.1%** | 95.4% | 95.7% | 95.5% |

### Stroke — Bleeding vs Ischemia (Model B)
- Accuracy: **97.30%** | AUC: **0.9728**
- Bleeding F1: 0.97 | Ischemia F1: 0.97

### Alzheimer's
- Overall accuracy: **93.04%**
- Moderate Impairment: Perfect recall (1.0000)
- VeryMild Impairment: 100% recall

### Evaluation Criteria
- ✅ **Grad-CAM** explainability — visualizing decision regions for clinical trust
- ✅ **Cross-validation** stability — robust performance across data folds
- ✅ **Inference benchmarking** — latency testing for real-time PACS integration
- ✅ **Calibration** — reliable confidence scores for clinical support

---

## Deployment

The **NeuroLens** web application was built with:

- **Frontend:** HTML5, CSS3, JavaScript — clean, clinical-grade UI
- **Backend:** Java — REST API serving the trained CNN models
- **Interface Features:**
  - Upload PNG / JPG / DICOM files via drag-and-drop
  - Three dedicated modules: **Tumor Detect**, **Stroke Detect**, **Alzheimer Detect**
  - Real-time prediction with confidence scores
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)
---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch |
| Model Architectures | ResNet-18/50, EfficientNet-B0, MobileNetV2 |
| Data Processing | torchvision, NumPy, PIL |
| Visualization | Matplotlib, Grad-CAM |
| Backend | Java |
| Frontend | HTML5, CSS3, JavaScript |
| Data Source | Kaggle |
| Methodology | CRISP-DM |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/neurolens.git
cd neurolens

# Install Python dependencies
pip install -r requirements.txt

# Train a model (example: brain tumor)
python train.py --disease tumor --model efficientnet_b0 --epochs 25

# Run the web application
# Start the Java backend
cd backend
mvn spring-boot:run

# Open frontend
open frontend/index.html
```

---



## Team

| Name | Role |
|------|------|
| **Yoldez Boubahri** | ML Engineer & Data Scientist |
| **Abir Bouhajja** | ML Engineer & Data Scientist |
| **Nour Nsiri** | ML Engineer & Data Scientist |

**Supervisor:** Mr. Seif Eddine Mejri  
**Institution:** ESSAI (École Supérieure des Sciences Appliquées et de l'Informatique)

---


---

<div align="center">
  <i>"Neurolens isn't just a project — it's a step toward a future where no patient waits too long, and no doctor stands alone."</i>
  <br><br>
  Made with ❤️ for better healthcare
</div>
