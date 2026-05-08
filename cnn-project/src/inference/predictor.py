"""
predictor.py
─────────────
Couche d'inférence unifiée pour tous les modèles.
Simplifie le chargement des modèles entraînés et les prédictions.

Exemple d'utilisation:
    from src.inference.predictor import StrokePredictor
    
    predictor = StrokePredictor(device='cuda')
    result = predictor.predict('image.jpg')
    print(result['message'])
"""

import torch
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.config import get_config, get_model_save_path


class BasePredictor(ABC):
    """Classe de base pour tous les predictors."""
    
    def __init__(self, disease: str, device: Optional[str] = None):
        self.disease = disease
        self.config = get_config(disease)
        self.device = torch.device(device or self.config.get('device', 'cpu'))
        self.model = None
        self.transform = None
    
    @abstractmethod
    def _build_model(self):
        """À implémenter dans les classes enfants."""
        pass
    
    @abstractmethod
    def predict(self, image_path: str) -> Dict[str, Any]:
        """À implémenter dans les classes enfants."""
        pass
    
    def _load_model_weights(self, model, model_type: str = 'a'):
        """Charge les poids entraînés."""
        weight_path = get_model_save_path(self.disease, model_type)
        if not weight_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weight_path}")
        
        model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model


class StrokePredictor(BasePredictor):
    """Predictor pour Stroke - Pipeline en cascade (Model A -> Model B)."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__('Stroke', device)
        self.model_a = None
        self.model_b = None
        self._build_model()
    
    def _build_model(self):
        """Construit et charge les deux modèles."""
        from src.model.model_stroke import build_model_a, build_model_b
        
        # Modèle A : détection (Normal vs Stroke)
        self.model_a = build_model_a()
        self.model_a = self._load_model_weights(self.model_a, 'a')
        
        # Modèle B : classification (Bleeding vs Ischemia)
        self.model_b = build_model_b()
        self.model_b = self._load_model_weights(self.model_b, 'b')
        
        # Transform
        from src.data.data_loader_stroke import val_transform
        self.transform = val_transform
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Prédiction complète du pipeline Stroke.
        
        Returns:
            {
                'disease': 'Stroke',
                'status': 'normal' | 'stroke' | 'low_confidence',
                'type': None | 'Bleeding' | 'Ischemia',
                'confidence_detection': float,
                'confidence_classification': float | None,
                'message': str
            }
        """
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Stage 1 : Detection (Model A)
        with torch.no_grad():
            logits_a = self.model_a(tensor)
            probs_a = torch.softmax(logits_a, dim=1)[0]
        
        conf_normal = probs_a[0].item()
        conf_stroke = probs_a[1].item()
        max_conf = max(conf_normal, conf_stroke)
        
        config_inference = self.config.get('inference', {})
        confidence_threshold = config_inference.get('confidence_threshold', 0.85)
        stroke_threshold = config_inference.get('stroke_threshold', 0.5)
        
        # Low confidence
        if max_conf < confidence_threshold:
            return {
                'disease': 'Stroke',
                'status': 'low_confidence',
                'type': None,
                'confidence_detection': max_conf,
                'confidence_classification': None,
                'message': f'⚠️ Low confidence ({max_conf:.1%}) - Consult radiologist'
            }
        
        # Normal
        if conf_stroke <= stroke_threshold:
            return {
                'disease': 'Stroke',
                'status': 'normal',
                'type': None,
                'confidence_detection': conf_normal,
                'confidence_classification': None,
                'message': f'✅ Normal Brain (confidence: {conf_normal:.1%})'
            }
        
        # Stage 2 : Classification (Model B)
        with torch.no_grad():
            logits_b = self.model_b(tensor)
            probs_b = torch.softmax(logits_b, dim=1)[0]
        
        pred_b = int(probs_b.argmax())
        conf_b = probs_b[pred_b].item()
        stroke_type = self.config['model_b']['classes'][pred_b]
        
        return {
            'disease': 'Stroke',
            'status': 'stroke',
            'type': stroke_type,
            'confidence_detection': conf_stroke,
            'confidence_classification': conf_b,
            'message': (f'🚨 Stroke Detected: {stroke_type} '
                       f'(detection: {conf_stroke:.1%}, '
                       f'classification: {conf_b:.1%})')
        }


class AlzheimerPredictor(BasePredictor):
    """Predictor pour Alzheimer."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__('Alzheimer', device)
        self._build_model()
    
    def _build_model(self):
        """Construit et charge le modèle."""
        from src.model.model_alzheimer import build_efficientnet_b0
        from src.data.data_loader_alzheimer import get_transforms
        
        self.model = build_efficientnet_b0()
        self.model = self._load_model_weights(self.model)
        
        _, self.transform = get_transforms()
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Prédiction pour Alzheimer."""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        pred_idx = int(probs.argmax())
        confidence = probs[pred_idx].item()
        class_name = self.config['model']['classes'][pred_idx]
        
        return {
            'disease': 'Alzheimer',
            'status': 'prediction',
            'classification': class_name,
            'confidence': confidence,
            'all_probs': {
                self.config['model']['classes'][i]: float(probs[i])
                for i in range(len(self.config['model']['classes']))
            },
            'message': f'{class_name}: {confidence:.1%} confidence'
        }


class BrainTumorPredictor(BasePredictor):
    """Predictor pour Brain Tumor - Pipeline en cascade (Model A -> Model B)."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__('Brain_Tumor', device)
        self.model_a = None
        self.model_b = None
        self._build_model()
    
    def _build_model(self):
        """Construit et charge les deux modèles."""
        from src.model.model import build_model_a, build_model_b
        from src.data.data_loader import val_transform
        
        # Modèle A : Binary (Tumor / No Tumor)
        self.model_a = build_model_a()
        self.model_a = self._load_model_weights(self.model_a, 'a')
        
        # Modèle B : 3-class (Glioma / Meningioma / Pituitary)
        self.model_b = build_model_b()
        self.model_b = self._load_model_weights(self.model_b, 'b')
        
        self.transform = val_transform
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Prédiction complète du pipeline Brain Tumor."""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Stage 1 : Binary detection (Model A)
        with torch.no_grad():
            logits_a = self.model_a(tensor)
            probs_a = torch.softmax(logits_a, dim=1)[0]
        
        conf_no_tumor = probs_a[0].item()
        conf_tumor = probs_a[1].item()
        
        # No tumor detected
        if conf_no_tumor > 0.5:
            return {
                'disease': 'Brain_Tumor',
                'status': 'no_tumor',
                'type': None,
                'confidence': conf_no_tumor,
                'message': f'✅ No Tumor (confidence: {conf_no_tumor:.1%})'
            }
        
        # Tumor detected - Stage 2: Classification
        with torch.no_grad():
            logits_b = self.model_b(tensor)
            probs_b = torch.softmax(logits_b, dim=1)[0]
        
        pred_b = int(probs_b.argmax())
        conf_b = probs_b[pred_b].item()
        tumor_type = self.config['model_b']['classes'][pred_b]
        
        return {
            'disease': 'Brain_Tumor',
            'status': 'tumor_detected',
            'type': tumor_type,
            'confidence_detection': conf_tumor,
            'confidence_classification': conf_b,
            'message': (f'🚨 Tumor Detected: {tumor_type} '
                       f'(detection: {conf_tumor:.1%}, '
                       f'classification: {conf_b:.1%})')
        }


# ────────────────────────────────────────────────────────────
# Factory pour créer le bon predictor
# ────────────────────────────────────────────────────────────

def get_predictor(disease: str, device: Optional[str] = None) -> BasePredictor:
    """
    Factory pour obtenir le bon predictor.
    
    Args:
        disease: 'Stroke', 'Alzheimer', ou 'Brain_Tumor'
        device: 'cuda' ou 'cpu'
    
    Returns:
        Instance du predictor approprié
    """
    predictors = {
        'Stroke': StrokePredictor,
        'Alzheimer': AlzheimerPredictor,
        'Brain_Tumor': BrainTumorPredictor
    }
    
    if disease not in predictors:
        raise ValueError(f"Disease '{disease}' not supported. "
                        f"Choose from: {list(predictors.keys())}")
    
    return predictors[disease](device=device)
