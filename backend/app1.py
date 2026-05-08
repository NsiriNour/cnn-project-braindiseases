"""
🧠 Brain MRI AI Backend (Production-Ready)
----------------------------------------
- Tumor Detection (MobileNetV2)
- Alzheimer Classification (ResNet50)
- Grad-CAM Visualization
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

# ───────────────── CONFIG ─────────────────
app = Flask(__name__)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONF_THRESHOLD = 0.6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TUMOR_CLASSES = ['No Tumor', 'Tumor Present']
ALZHEIMER_CLASSES = ['NonDemented', 'Demented', 'ModerateDemented']

# ───────────────── MODELS ─────────────────

def build_tumor_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def build_alzheimer_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3)  # 3 classes
    )
    return model

# ───────────────── LOAD MODELS (STARTUP) ─────────────────

print("🔄 Loading models...")

try:
    tumor_model = build_tumor_model()
    if os.path.exists("model_a_binary.pth"):
        tumor_model.load_state_dict(torch.load("model_a_binary.pth", map_location=DEVICE), strict=True)
        print("✅ Tumor model loaded")
    tumor_model.to(DEVICE).eval()
except Exception as e:
    print(f"⚠️ Tumor model error: {e}")
    tumor_model = None

try:
    alzheimer_model = build_alzheimer_model()
    if os.path.exists("model_b_3class.pth"):
        alzheimer_model.load_state_dict(torch.load("model_b_3class.pth", map_location=DEVICE), strict=True)
        print("✅ Alzheimer model loaded")
    alzheimer_model.to(DEVICE).eval()
except Exception as e:
    print(f"⚠️ Alzheimer model error: {e}")
    alzheimer_model = None

print("✅ Models initialized!")

# ───────────────── PREPROCESS ─────────────────

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def preprocess(file):
    img = Image.open(file).convert('RGB')
    return transform(img).unsqueeze(0), img

# ───────────────── GRADCAM ─────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.grad = None
        self.act = None

        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0]))

        self.model = model

    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.act).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().numpy()

def make_heatmap(img, cam):
    cam = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(224,224)).squeeze().numpy()

    img_np = np.array(img.resize((224,224))) / 255.0
    heat = colormap.jet(cam)[:,:,:3]

    overlay = (0.6 * img_np + 0.4 * heat).clip(0,1)

    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(img_np); ax[0].axis('off')
    ax[1].imshow(overlay); ax[1].axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return base64.b64encode(buf.read()).decode()

# ───────────────── ROUTES ─────────────────

def handle_prediction(model, classes, file, target_layer):
    tensor, img = preprocess(file)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])

    if conf < CONF_THRESHOLD:
        return {'success': False, 'message': 'Low confidence', 'confidence': conf}

    # GradCAM
    try:
        cam = GradCAM(model, target_layer).generate(tensor, pred)
        heatmap = make_heatmap(img, cam)
    except:
        heatmap = None

    return {
        'success': True,
        'prediction': classes[pred],
        'confidence': conf,
        'heatmap': heatmap
    }

@app.route('/predict/tumor', methods=['POST'])
def predict_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    result = handle_prediction(
        tumor_model,
        TUMOR_CLASSES,
        request.files['image'],
        tumor_model.features[-1]
    )

    return jsonify(result)

@app.route('/predict/alzheimer', methods=['POST'])
def predict_alzheimer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    result = handle_prediction(
        alzheimer_model,
        ALZHEIMER_CLASSES,
        request.files['image'],
        alzheimer_model.layer4[-1]
    )

    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE)
    })

# ───────────────── RUN ─────────────────

if __name__ == '__main__':
    print("⚠️  app1.py is deprecated. Use python app.py instead.")
    print("🚀 Server running at http://localhost:5000")
    app.run(debug=True)