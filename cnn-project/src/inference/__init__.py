"""
inference package
──────────────────
Module d'inférence unifié pour tous les modèles.
Facilite le chargement et l'utilisation des modèles entraînés.

Exemple:
    from src.inference import get_predictor
    
    predictor = get_predictor('Stroke', device='cuda')
    result = predictor.predict('image.jpg')
"""

from .predictor import (
    StrokePredictor,
    AlzheimerPredictor,
    BrainTumorPredictor,
    get_predictor
)

__all__ = [
    'StrokePredictor',
    'AlzheimerPredictor',
    'BrainTumorPredictor',
    'get_predictor'
]
