# NeuroLens - AI Medical Imaging Analysis Platform

A deep learning system for automated neurological imaging classification with a Flask backend and a browser-based frontend.

## 🎯 Project Overview

NeuroLens provides:
- **Brain tumor detection** with binary and tumor subtype classification
- **Alzheimer's disease staging** using an EfficientNet B0-based model
- **Stroke detection** with binary and stroke-type predictions
- **Integrated web frontend** for image upload and inference
- **Grad-CAM heatmap support** for model explainability

## 📂 Workspace Structure

```
cnn/
├── backend/
│   ├── app2.py                 # Primary Flask inference server
│   ├── app1.py                 # Alternative backend implementation
│   ├── requirements.txt        # Backend dependencies
│   ├── models/                 # Saved PyTorch model weights
│   │   ├── efficientnet_b0_alzheimer.pt
│   │   ├── model_a_binary.pth
│   │   ├── model_a_stroke.pth
│   │   ├── model_b_3class.pth
│   │   └── model_b_stroketype.pth
│   ├── uploads/                # Temporary uploaded files
│   ├── pg.ipynb                # Prototype notebook
│   ├── test_upload.py          # Upload endpoint tester
│   └── test_write.txt          # Test file output sample
└── frontend/
    ├── index.html              # Main upload UI
    ├── brainTumor.html         # Brain tumor results page
    ├── alzheimer.html          # Alzheimer's results page
    ├── brainStroke.html        # Stroke prediction page
    ├── index.css               # Shared page styles
    └── tumor.js                # Frontend logic for uploads/results
```

## 🧠 Backend Overview

The current Flask server is implemented in `cnn/backend/app2.py` and exposes the following endpoints:

- `POST /predict/tumor`
- `POST /predict/alzheimer`
- `POST /predict/stroke`
- `GET /health`

The backend loads PyTorch models from `cnn/backend/models/` and does image preprocessing with torchvision transforms.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.x
- Flask
- Flask-CORS
- Pillow
- NumPy
- torchvision

### Backend Setup

```bash
cd cnn/backend
pip install -r requirements.txt
python app2.py
```

The Flask server starts on `http://127.0.0.1:5000`.

### Frontend Setup

Open `cnn/frontend/index.html` in your browser, or serve the frontend directory with a local HTTP server:

```bash
cd cnn/frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`.

## 🔌 API Details

### Tumor Prediction

`POST /predict/tumor`
- Form field: `image`
- Returns:
  - `prediction`
  - `class_index`
  - `confidence`
  - `probabilities`
  - `tumor_type`
  - `tumor_type_confidence`
  - `tumor_type_probabilities`
  - `heatmap`

### Alzheimer Prediction

`POST /predict/alzheimer`
- Form field: `image`
- Returns:
  - `prediction`
  - `class_index`
  - `confidence`
  - `probabilities`
  - `heatmap`

### Stroke Prediction

`POST /predict/stroke`
- Form field: `image`
- Returns:
  - `prediction`
  - `class_index`
  - `confidence`
  - `probabilities`
  - `stroke_type`
  - `stroke_type_confidence`
  - `stroke_type_probabilities`
  - `heatmap`

### Health Check

`GET /health`
- Returns server status, device, and model load state.

## 📊 Image Preprocessing

Input images are processed using:
- Resize to 256
- Center crop to 224×224
- RGB conversion
- ImageNet normalization

## 📌 Notes

- The server detects CUDA automatically and uses GPU when available.
- `cnn/backend/app2.py` is the current inference entrypoint.
- Model weights must exist in `cnn/backend/models/` or the backend directory.

## 📦 Backend Dependencies

`cnn/backend/requirements.txt` includes:
- Flask==2.3.3
- Flask-CORS==4.0.0
- torch==2.6.0
- torchvision==0.21.0
- Pillow==10.0.0
- numpy==1.24.3
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
