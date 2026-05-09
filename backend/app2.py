import os
import io
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as colormap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', '..', '..', 'cnn-project'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path=''
)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BINARY_CLASSES = ['No Tumor', 'Tumor Present']
TUMOR_TYPE_CLASSES = ['glioma', 'meningioma', 'pituitary']
ALZHEIMER_CLASSES = [
    'Mild Impairment',
    'Moderate Impairment',
    'No Impairment',
    'Very Mild Impairment',
]
STROKE_BINARY_CLASSES = ['No Stroke', 'Stroke Present']
STROKE_TYPE_CLASSES = ['Bleeding', 'Ischemia']


def build_tumor_binary_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2)
    )
    return model


def build_tumor_type_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(TUMOR_TYPE_CLASSES))
    )
    return model


def build_alzheimer_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 4)
    )
    return model


def build_stroke_binary_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2)
    )
    return model


def build_stroke_type_model():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(STROKE_TYPE_CLASSES))
    )
    return model


def load_model(path, build_fn):
    if not os.path.exists(path):
        return None
    try:
        loaded = torch.load(path, map_location=DEVICE)
        if isinstance(loaded, dict):
            # It's a state_dict
            model = build_fn()
            model.load_state_dict(loaded, strict=False)
        else:
            # It's a full model
            model = loaded
        return model.to(DEVICE).eval()
    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")
        return None


tumor_binary_path = os.path.join(MODELS_DIR, 'model_a_binary.pth')
if not os.path.exists(tumor_binary_path):
    tumor_binary_path = os.path.join(BASE_DIR, 'model_a_binary.pth')

tumor_binary_model = load_model(tumor_binary_path, build_tumor_binary_model)

tumor_type_path = os.path.join(MODELS_DIR, 'model_b_3class.pth')
if not os.path.exists(tumor_type_path):
    tumor_type_path = os.path.join(BASE_DIR, 'model_b_3class.pth')

tumor_type_model = load_model(tumor_type_path, build_tumor_type_model)

alzheimer_path = os.path.join(MODELS_DIR, 'efficientnet_b0_alzheimer.pt')
if not os.path.exists(alzheimer_path):
    alzheimer_path = os.path.join(BASE_DIR, 'efficientnet_b0_alzheimer.pt')

alzheimer_model = load_model(alzheimer_path, build_alzheimer_model)

stroke_binary_path = os.path.join(MODELS_DIR, 'model_a_stroke.pth')
if not os.path.exists(stroke_binary_path):
    stroke_binary_path = os.path.join(BASE_DIR, 'model_a_stroke.pth')

stroke_binary_model = load_model(stroke_binary_path, build_stroke_binary_model)

stroke_type_path = os.path.join(MODELS_DIR, 'model_b_stroketype.pth')
if not os.path.exists(stroke_type_path):
    stroke_type_path = os.path.join(BASE_DIR, 'model_b_stroketype.pth')

stroke_type_model = load_model(stroke_type_path, build_stroke_type_model)


def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(image).unsqueeze(0), image


def generate_gradcam_overlay(original_img, cam_array):
    cam_up = F.interpolate(
        torch.tensor(cam_array).unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    img_np = np.array(original_img.resize((224, 224))) / 255.0
    heat = colormap.jet(cam_up)[:, :, :3]
    overlay = (0.55 * img_np + 0.45 * heat).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, image_tensor, device):
        self.model.eval()
        img = image_tensor.to(device)
        img.requires_grad = True
        output = self.model(img)
        class_idx = int(output.argmax(dim=1).item())
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/tumor', methods=['POST'])
def predict_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img_tensor, original_img = preprocess_image(file)

    global tumor_binary_model, tumor_type_model
    if tumor_binary_model is None:
        tumor_binary_model = load_model(os.path.join(MODELS_DIR, 'model_a_binary.pth'), build_tumor_binary_model)
        if tumor_binary_model is None:
            return jsonify({'error': 'Tumor binary model is not available'}), 500

    with torch.no_grad():
        output = tumor_binary_model(img_tensor.to(DEVICE))
        probabilities = F.softmax(output, dim=1)[0]
        binary_idx = int(output.argmax(dim=1).item())

    prediction_text = BINARY_CLASSES[binary_idx]
    tumor_type = None
    tumor_type_probability = None
    tumor_type_probabilities = {}

    if binary_idx == 1:
        if tumor_type_model is None:
            tumor_type_model = load_model(tumor_type_path, build_tumor_type_model)

        if tumor_type_model is not None:
            with torch.no_grad():
                type_output = tumor_type_model(img_tensor.to(DEVICE))
                type_probs = F.softmax(type_output, dim=1)[0]
                type_idx = int(type_output.argmax(dim=1).item())
                tumor_type = TUMOR_TYPE_CLASSES[type_idx]
                tumor_type_probability = float(type_probs[type_idx])
                tumor_type_probabilities = {TUMOR_TYPE_CLASSES[i]: float(type_probs[i]) for i in range(len(TUMOR_TYPE_CLASSES))}
                prediction_text = f"Tumor Present — {tumor_type}"
        else:
            prediction_text = 'Tumor Present'

    heatmap_img = None
    try:
        target_layer = tumor_binary_model.features[-1]
        gradcam = GradCAM(tumor_binary_model, target_layer)
        cam, _ = gradcam.generate(img_tensor, DEVICE)
        heatmap_img = generate_gradcam_overlay(original_img, cam)
    except Exception as e:
        print(f"GradCAM error: {e}")

    return jsonify({
        'prediction': prediction_text,
        'class_index': binary_idx,
        'confidence': float(probabilities[binary_idx]),
        'probabilities': {BINARY_CLASSES[i]: float(probabilities[i]) for i in range(len(BINARY_CLASSES))},
        'tumor_type': tumor_type,
        'tumor_type_confidence': tumor_type_probability,
        'tumor_type_probabilities': tumor_type_probabilities,
        'heatmap': heatmap_img
    })


@app.route('/predict/alzheimer', methods=['POST'])
def predict_alzheimer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img_tensor, original_img = preprocess_image(file)

    global alzheimer_model
    if alzheimer_model is None:
        alzheimer_model = load_model(os.path.join(MODELS_DIR, 'efficientnet_b0_alzheimer.pt'), build_alzheimer_model)
        if alzheimer_model is None:
            return jsonify({'error': 'Alzheimer model is not available'}), 500

    with torch.no_grad():
        output = alzheimer_model(img_tensor.to(DEVICE))
        probabilities = F.softmax(output, dim=1)[0]
        raw_idx = int(output.argmax(dim=1).item())

    confidence = float(probabilities[raw_idx])
    class_probabilities = {
        ALZHEIMER_CLASSES[i]: float(probabilities[i])
        for i in range(len(ALZHEIMER_CLASSES))
    }

    heatmap_img = None
    try:
        target_layer = alzheimer_model.features[-1]
        gradcam = GradCAM(alzheimer_model, target_layer)
        cam, _ = gradcam.generate(img_tensor, DEVICE)
        heatmap_img = generate_gradcam_overlay(original_img, cam)
    except Exception as e:
        print(f"GradCAM error: {e}")

    return jsonify({
        'prediction': ALZHEIMER_CLASSES[raw_idx],
        'class_index': raw_idx,
        'confidence': confidence,
        'probabilities': class_probabilities,
        'heatmap': heatmap_img
    })


@app.route('/predict/stroke', methods=['POST'])
def predict_stroke():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img_tensor, original_img = preprocess_image(file)

    global stroke_binary_model, stroke_type_model
    if stroke_binary_model is None:
        stroke_binary_model = load_model(os.path.join(MODELS_DIR, 'model_a_stroke.pth'), build_stroke_binary_model)
        if stroke_binary_model is None:
            return jsonify({'error': 'Stroke binary model is not available'}), 500

    with torch.no_grad():
        output = stroke_binary_model(img_tensor.to(DEVICE))
        probabilities = F.softmax(output, dim=1)[0]
        binary_idx = int(output.argmax(dim=1).item())

    prediction_text = STROKE_BINARY_CLASSES[binary_idx]
    stroke_type = None
    stroke_type_probability = None
    stroke_type_probabilities = {}

    if binary_idx == 1:
        if stroke_type_model is None:
            stroke_type_model = load_model(stroke_type_path, build_stroke_type_model)

        if stroke_type_model is not None:
            with torch.no_grad():
                type_output = stroke_type_model(img_tensor.to(DEVICE))
                type_probs = F.softmax(type_output, dim=1)[0]
                type_idx = int(type_output.argmax(dim=1).item())
                stroke_type = STROKE_TYPE_CLASSES[type_idx]
                stroke_type_probability = float(type_probs[type_idx])
                stroke_type_probabilities = {STROKE_TYPE_CLASSES[i]: float(type_probs[i]) for i in range(len(STROKE_TYPE_CLASSES))}
                prediction_text = f"Stroke Present — {stroke_type}"
        else:
            prediction_text = 'Stroke Present'

    heatmap_img = None
    try:
        target_layer = stroke_binary_model.features[-1]
        gradcam = GradCAM(stroke_binary_model, target_layer)
        cam, _ = gradcam.generate(img_tensor, DEVICE)
        heatmap_img = generate_gradcam_overlay(original_img, cam)
    except Exception as e:
        print(f"GradCAM error: {e}")

    return jsonify({
        'prediction': prediction_text,
        'class_index': binary_idx,
        'confidence': float(probabilities[binary_idx]),
        'probabilities': {STROKE_BINARY_CLASSES[i]: float(probabilities[i]) for i in range(len(STROKE_BINARY_CLASSES))},
        'stroke_type': stroke_type,
        'stroke_type_confidence': stroke_type_probability,
        'stroke_type_probabilities': stroke_type_probabilities,
        'heatmap': heatmap_img
    })


@app.route('/predict/all', methods=['POST'])
def predict_all():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img_tensor, original_img = preprocess_image(file)

    results = {}

    # Tumor prediction
    try:
        global tumor_binary_model, tumor_type_model
        if tumor_binary_model is None:
            tumor_binary_model = load_model(tumor_binary_path, build_tumor_binary_model)
        
        if tumor_binary_model is not None:
            with torch.no_grad():
                output = tumor_binary_model(img_tensor.to(DEVICE))
                probabilities = F.softmax(output, dim=1)[0]
                binary_idx = int(output.argmax(dim=1).item())

            tumor_probabilities = {BINARY_CLASSES[i]: float(probabilities[i]) for i in range(len(BINARY_CLASSES))}
            tumor_type = None
            tumor_type_probabilities = {}

            if binary_idx == 1:
                if tumor_type_model is None:
                    tumor_type_model = load_model(tumor_type_path, build_tumor_type_model)
                if tumor_type_model is not None:
                    with torch.no_grad():
                        type_output = tumor_type_model(img_tensor.to(DEVICE))
                        type_probs = F.softmax(type_output, dim=1)[0]
                        type_idx = int(type_output.argmax(dim=1).item())
                        tumor_type = TUMOR_TYPE_CLASSES[type_idx]
                        tumor_type_probabilities = {TUMOR_TYPE_CLASSES[i]: float(type_probs[i]) for i in range(len(TUMOR_TYPE_CLASSES))}

            # Calculate sick probability correctly
            tumor_sick_prob = 1.0 - float(probabilities[0])  # 1 - prob of 'No Tumor'
            
            tumor_prediction = BINARY_CLASSES[binary_idx]
            if binary_idx == 1 and tumor_type:
                tumor_prediction = f"Tumor Present — {tumor_type}"

            results['tumor'] = {
                'prediction': tumor_prediction,
                'class_index': binary_idx,
                'confidence': float(probabilities[binary_idx]),
                'probabilities': tumor_probabilities,
                'tumor_type': tumor_type,
                'tumor_type_probabilities': tumor_type_probabilities,
                'sick_probability': tumor_sick_prob
            }
        else:
            results['tumor'] = {'error': 'Tumor model not available'}
    except Exception as e:
        results['tumor'] = {'error': str(e)}

    # Alzheimer prediction
    try:
        global alzheimer_model
        if alzheimer_model is None:
            alzheimer_model = load_model(alzheimer_path, build_alzheimer_model)
        
        if alzheimer_model is not None:
            with torch.no_grad():
                output = alzheimer_model(img_tensor.to(DEVICE))
                probabilities = F.softmax(output, dim=1)[0]
                idx = int(output.argmax(dim=1).item())

            alz_probabilities = {ALZHEIMER_CLASSES[i]: float(probabilities[i]) for i in range(len(ALZHEIMER_CLASSES))}
            # 'No Impairment' is index 2
            no_impairment_prob = float(probabilities[2])
            # Sick probability = 1 - probability of 'No Impairment' (having any impairment)
            alz_sick_prob = 1.0 - no_impairment_prob

            results['alzheimer'] = {
                'prediction': ALZHEIMER_CLASSES[idx],
                'class_index': idx,
                'confidence': float(probabilities[idx]),
                'probabilities': alz_probabilities,
                'sick_probability': alz_sick_prob
            }
        else:
            results['alzheimer'] = {'error': 'Alzheimer model not available'}
    except Exception as e:
        results['alzheimer'] = {'error': str(e)}

    # Stroke prediction
    try:
        global stroke_binary_model, stroke_type_model
        if stroke_binary_model is None:
            stroke_binary_model = load_model(stroke_binary_path, build_stroke_binary_model)
        
        if stroke_binary_model is not None:
            with torch.no_grad():
                output = stroke_binary_model(img_tensor.to(DEVICE))
                probabilities = F.softmax(output, dim=1)[0]
                binary_idx = int(output.argmax(dim=1).item())

            stroke_probabilities = {STROKE_BINARY_CLASSES[i]: float(probabilities[i]) for i in range(len(STROKE_BINARY_CLASSES))}
            stroke_type = None
            stroke_type_probabilities = {}

            if binary_idx == 1:
                if stroke_type_model is None:
                    stroke_type_model = load_model(stroke_type_path, build_stroke_type_model)
                if stroke_type_model is not None:
                    with torch.no_grad():
                        type_output = stroke_type_model(img_tensor.to(DEVICE))
                        type_probs = F.softmax(type_output, dim=1)[0]
                        type_idx = int(type_output.argmax(dim=1).item())
                        stroke_type = STROKE_TYPE_CLASSES[type_idx]
                        stroke_type_probabilities = {STROKE_TYPE_CLASSES[i]: float(type_probs[i]) for i in range(len(STROKE_TYPE_CLASSES))}

            # Calculate sick probability correctly
            stroke_sick_prob = 1.0 - float(probabilities[0])  # 1 - prob of 'No Stroke'
            
            stroke_prediction = STROKE_BINARY_CLASSES[binary_idx]
            if binary_idx == 1 and stroke_type:
                stroke_prediction = f"Stroke Present — {stroke_type}"

            results['stroke'] = {
                'prediction': stroke_prediction,
                'class_index': binary_idx,
                'confidence': float(probabilities[binary_idx]),
                'probabilities': stroke_probabilities,
                'stroke_type': stroke_type,
                'stroke_type_probabilities': stroke_type_probabilities,
                'sick_probability': stroke_sick_prob
            }
        else:
            results['stroke'] = {'error': 'Stroke model not available'}
    except Exception as e:
        results['stroke'] = {'error': str(e)}

    # Generate heatmaps for all models
    try:
        # Tumor heatmap
        if tumor_binary_model is not None and 'tumor' in results and 'error' not in results['tumor']:
            target_layer = tumor_binary_model.features[-1]
            gradcam = GradCAM(tumor_binary_model, target_layer)
            cam, _ = gradcam.generate(img_tensor, DEVICE)
            tumor_heatmap = generate_gradcam_overlay(original_img, cam)
            results['tumor']['heatmap'] = tumor_heatmap
        
        # Alzheimer heatmap
        if alzheimer_model is not None and 'alzheimer' in results and 'error' not in results['alzheimer']:
            target_layer = alzheimer_model.features[-1]
            gradcam = GradCAM(alzheimer_model, target_layer)
            cam, _ = gradcam.generate(img_tensor, DEVICE)
            alzheimer_heatmap = generate_gradcam_overlay(original_img, cam)
            results['alzheimer']['heatmap'] = alzheimer_heatmap
            
        # Stroke heatmap
        if stroke_binary_model is not None and 'stroke' in results and 'error' not in results['stroke']:
            target_layer = stroke_binary_model.features[-1]
            gradcam = GradCAM(stroke_binary_model, target_layer)
            cam, _ = gradcam.generate(img_tensor, DEVICE)
            stroke_heatmap = generate_gradcam_overlay(original_img, cam)
            results['stroke']['heatmap'] = stroke_heatmap
            
    except Exception as e:
        print(f"GradCAM error: {e}")

    return jsonify({
        'results': results
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'models_loaded': {
            'tumor_binary': tumor_binary_model is not None,
            'tumor_type': tumor_type_model is not None,
            'alzheimer': alzheimer_model is not None,
            'stroke_binary': stroke_binary_model is not None,
            'stroke_type': stroke_type_model is not None
        }
    })


if __name__ == '__main__':
    print('🧠 Brain MRI Classification Server')
    print(f'Device: {DEVICE}')
    print(f'Models loaded: tumor_binary={tumor_binary_model is not None}, tumor_type={tumor_type_model is not None}, alzheimer={alzheimer_model is not None}, stroke_binary={stroke_binary_model is not None}, stroke_type={stroke_type_model is not None}')
    print(f'Alzheimer model path: {alzheimer_path}')
    app.run(debug=True, host='127.0.0.1', port=5000)

