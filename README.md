# NeuroLens - AI Medical Imaging Analysis Platform

A comprehensive deep learning system for automated detection and classification of neurological conditions using medical imaging. This platform includes brain tumor detection, Alzheimer's disease classification, and an interactive web application for clinical decision support.

## 🎯 Project Overview

NeuroLens combines state-of-the-art CNN models with a user-friendly web interface to provide:
- **Brain Tumor Classification**: Binary (tumor/healthy) and multi-class (glioma/meningioma/pituitary/none) detection
- **Alzheimer's Disease Staging**: Classification into No Impairment, Very Mild, Mild, and Moderate stages
- **Real-time Inference**: GPU-accelerated prediction with confidence scores
- **Clinical Interface**: Web application for easy image upload and result visualization

## 📂 Project Structure

```
NeuroLens/
│
├── cnn/                              # Web Application (Flask + Frontend)
│   ├── backend/
│   │   ├── app.py                   # Main Flask application
│   │   ├── app1.py                  # Alternative backend implementation
│   │   ├── requirements.txt         # Python dependencies
│   │   ├── model_a_binary.pth       # Binary tumor classification model
│   │   ├── model_b_3class.pth       # Multi-class tumor model
│   │   ├── uploads/                 # Temporary image storage
│   │   └── test_upload.py           # Upload testing script
│   └── frontend/
│       ├── index.html               # Main upload interface
│       ├── brainTumor.html          # Brain tumor results page
│       ├── alzheimer.html           # Alzheimer's results page
│       ├── brainStroke.html         # Stroke detection page
│       ├── tumor.js                 # Brain tumor JavaScript logic
│       └── index.css                # Shared styles
│
└── cnn-project/                      # Training & Model Development
    ├── train.ipynb                  # Brain tumor training notebook
    ├── train_alzheimer.ipynb        # Alzheimer's training notebook
    ├── pg.ipynb                     # Data exploration notebook
    ├── weights_scratch_a.pt         # Training weights checkpoint
    │
    ├── models/
    │   ├── model.py                 # Brain tumor CNN architecture
    │   ├── model_alzheimer.py       # Alzheimer's CNN architecture
    │   ├── train_utils.py           # Brain tumor training utilities
    │   ├── train_utils_alzheimer.py # Alzheimer's training utilities
    │   ├── model_a_binary.pth       # Trained binary classifier
    │   └── efficientnet_b0_alzheimer.pt # EfficientNet Alzheimer's model
    │
    ├── data/
    │   ├── data_loader.py           # Brain tumor data loading
    │   ├── data_loader_alzheimer.py # Alzheimer's data loading
    │   └── DATASET/
    │       ├── alz/                 # Alzheimer's dataset (train/test split)
    │       │   └── Combined Dataset/
    │       │       ├── train/       # Training images
    │       │       │   ├── No Impairment/
    │       │       │   ├── Very Mild Impairment/
    │       │       │   ├── Mild Impairment/
    │       │       │   └── Moderate Impairment/
    │       │       └── test/        # Test images (same classes)
    │       └── classification/      # Brain tumor dataset
    │           ├── Training/
    │           │   ├── glioma/
    │           │   ├── meningioma/
    │           │   ├── notumor/
    │           │   └── pituitary/
    │           └── Testing/
    │               ├── glioma/
    │               ├── meningioma/
    │               ├── notumor/
    │               └── pituitary/
    │
    ├── results/
    │   └── alzheimer/
    │       └── results_alzheimer.json  # Training results & metrics
    │
    └── utils/
        └── helpers.py               # Utility functions
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- Flask
- OpenCV
- PIL/Pillow

### Backend Setup

```bash
# Navigate to backend directory
cd cnn/backend

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

The backend will be available at `http://localhost:5000`

### Frontend Setup

**Option 1: Direct Browser Access**
```bash
# Simply open the file in your browser
start cnn/frontend/index.html
```

**Option 2: Local Web Server (Recommended)**
```bash
# Using Python 3
python -m http.server 8000 --directory cnn/frontend

# Visit: http://localhost:8000
```

## 🧠 Model Architecture & Specifications

### Brain Tumor Detection

**Binary Classification Model (model_a_binary.pth)**
- **Task**: Healthy vs Tumor
- **Input**: 224×224 RGB images
- **Architecture**:
  - 3 Convolutional blocks with ReLU activation
  - Adaptive average pooling
  - 2-layer classifier head with dropout
- **Dataset**: Brain tumor classification dataset (4 tumor types)
- **Performance**: Binary classification with confidence scores

**Multi-class Classification Model (model_b_3class.pth)**
- **Task**: Tumor type classification
- **Classes**: Glioma, Meningioma, Pituitary tumor, No tumor
- **Architecture**: Same as binary model
- **Input Preprocessing**: 
  - Resize to 224×224
  - Convert to RGB (3 channels)
  - Normalize with ImageNet statistics

### Alzheimer's Disease Classification (EfficientNet B0)

- **Task**: Disease staging classification
- **Classes**: No Impairment, Very Mild, Mild, Moderate
- **Architecture**: EfficientNet-B0 backbone
- **Dataset**: Combined Alzheimer's MRI dataset
- **Model File**: `efficientnet_b0_alzheimer.pt`
- **Results**: Saved in `cnn-project/results/alzheimer/results_alzheimer.json`

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect-tumor` | Brain tumor detection using binary & multi-class models |
| POST | `/detect-alzheimer` | Alzheimer's disease classification |
| POST | `/detect-stroke` | Stroke detection (planned) |
| GET | `/health` | Server health check |

### Request Format (Multipart Form Data)
```
POST /detect-tumor
- file: [image_file]
- Returns: {
    "binary_prediction": "tumor/healthy",
    "binary_confidence": 0.95,
    "class_prediction": "glioma",
    "class_confidence": 0.92,
    "class_probabilities": {...}
  }
```

## 📊 Image Processing Pipeline

All input images undergo automatic processing:

1. **Format Conversion**: Ensure RGB format (3 channels)
2. **Resizing**: Scale to 224×224 pixels using interpolation
3. **Normalization**: Apply ImageNet normalization:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
4. **Tensor Conversion**: Convert to PyTorch tensors
5. **Inference**: Feed to GPU/CPU models

## ✨ Features

### Frontend
- ✅ Drag & drop file upload interface
- ✅ Click-to-browse file selector
- ✅ Real-time inference with loading indicators
- ✅ Confidence score display
- ✅ Probability breakdown visualization
- ✅ Patient/Study metadata input
- ✅ Report generation & download
- ✅ Multi-condition support (tumor, Alzheimer's, stroke)

### Backend
- ✅ GPU support (automatic CUDA detection)
- ✅ CORS enabled for cross-origin requests
- ✅ Error handling & validation
- ✅ Model loading optimization
- ✅ Batch processing capability
- ✅ Health check endpoint
- ✅ Request logging

## 🔧 Training Models

### Training Brain Tumor Classifier

```bash
# Navigate to project directory
cd cnn-project

# Run training notebook
jupyter notebook train.ipynb
```

Configuration in `train.ipynb`:
- Dataset path: `src/data/DATASET/classification/`
- Model architecture: Custom CNN in `src/model/model.py`
- Training utilities: `src/model/train_utils.py`
- Output: Trained weights saved as `.pth` files

### Training Alzheimer's Classifier

```bash
jupyter notebook train_alzheimer.ipynb
```

Configuration:
- Dataset path: `src/data/DATASET/alz/Combined Dataset/`
- Model: EfficientNet-B0 in `src/model/model_alzheimer.py`
- Data loading: `src/data/data_loader_alzheimer.py`
- Training utilities: `src/model/train_utils_alzheimer.py`
- Results saved to: `results/alzheimer/results_alzheimer.json`

## 📈 Model Performance

### Alzheimer's Classification Results
See `cnn-project/results/alzheimer/results_alzheimer.json` for:
- Training/Validation accuracy
- Loss curves
- Per-class metrics
- Confusion matrices
- Model checkpoints

## 🐛 Troubleshooting

### CORS Errors
Ensure the Flask backend is running and CORS is properly configured in `app.py`.

### Model Loading Issues
- Verify model files exist in the correct directories
- Check PyTorch version compatibility
- Ensure GPU/CUDA drivers are installed if using GPU

### Image Upload Failures
- Check file format (supported: JPG, PNG, JPEG)
- Verify file size constraints
- Ensure `uploads/` directory has write permissions

### GPU Out of Memory
- Reduce batch size in training
- Use smaller model variants
- Enable gradient checkpointing

## 📋 Dataset Information

### Brain Tumor Classification
- **Location**: `cnn-project/src/data/DATASET/classification/`
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Split**: Training/Testing directories
- **Format**: PNG/JPG images

### Alzheimer's MRI Dataset
- **Location**: `cnn-project/src/data/DATASET/alz/Combined Dataset/`
- **Classes**: No Impairment, Very Mild, Mild, Moderate
- **Split**: Train/Test with balanced classes
- **Resolution**: Standardized MRI images

## 🔐 Security Notes

- Uploaded images are stored temporarily in `cnn/backend/uploads/`
- Consider implementing image cleanup routines for production
- Validate all file uploads server-side
- Use HTTPS in production deployment

## 📦 Dependencies

See `cnn/backend/requirements.txt` for complete list. Key packages:
- Flask
- PyTorch
- OpenCV (cv2)
- Pillow
- NumPy
- scikit-learn

## 🚢 Deployment

### Development
```bash
python cnn/backend/app.py  # Flask debug mode
```

### Production
Use a production WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 cnn.backend.app:app
```

## 📝 License

This project is intended for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Models are properly validated on test sets
2. Code follows PEP 8 standards
3. New features include documentation
4. Training results are reproducible

## 📧 Support

For issues or questions, please check:
- Model training notebooks for implementation details
- `utils/helpers.py` for utility functions
- API endpoint documentation in `backend/app.py`

---

**Last Updated**: May 2026  
**Status**: Active Development

## Customization

### Change Model Architecture
Edit the `SimpleCNN` class in `app.py` to match your actual model structure.

### Adjust Preprocessing
Modify the `transform` variable in `app.py` to match your training preprocessing.

### Add New Detection Types
1. Create a new results page (e.g., `newCondition.html`)
2. Add a new endpoint in `app.py` (e.g., `/detect-newcondition`)
3. Add navigation button in `index.html`

## Troubleshooting

### Models Not Loading
- Ensure `.pth` files are in the backend directory
- Check model file names match those in `app.py`

### CORS Errors
- Ensure Flask-CORS is installed and enabled in `app.py`

### Image Upload Fails
- Verify image format is PNG, JPG, or JPEG
- Check file size is reasonable

### GPU Not Detected
- Install CUDA and cuDNN
- Update PyTorch with GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## API Response Format

Successful detection response:
```json
{
    "success": true,
    "diagnosis": "Brain Tumor",
    "confidence": 94.2,
    "status": "danger",
    "probabilities": {
        "Healthy": 5.8,
        "Brain Tumor": 94.2
    },
    "model_type": "Binary Classification"
}
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- Modern web browser

## License

This project is for educational and medical research purposes.
