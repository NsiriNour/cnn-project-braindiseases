"""
output_manager.py
──────────────────
Gère la sauvegarde structurée de tous les outputs :
- Historiques d'entraînement (JSON)
- Courbes d'entraînement (PNG)
- Matrices de confusion (PNG)
- Métriques et rapports
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import get_paths, config_manager


class OutputManager:
    """Gère la sauvegarde structurée des outputs."""
    
    def __init__(self, disease: str):
        self.disease = disease
        self.paths = get_paths(disease)
        self.config = config_manager.load_config(disease)
    
    def save_training_history(self, history: Dict, model_type: str = 'a') -> Path:
        """
        Sauvegarde l'historique d'entraînement en JSON.
        
        Example:
            save_path = manager.save_training_history(history, 'a')
            # Sauvegarde dans: results/Stroke/history/history_a.json
        """
        if self.disease == 'Alzheimer':
            history_path = self.paths['history'] / 'history.json'
        else:
            history_path = self.paths['history'] / f'history_{model_type}.json'
        
        # Convertir les tensors en floats si nécessaire
        history_clean = {}
        for key, values in history.items():
            if isinstance(values, list):
                history_clean[key] = [float(v) if isinstance(v, torch.Tensor) else v 
                                     for v in values]
            else:
                history_clean[key] = values
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_clean, f, indent=2)
        
        print(f"✅ History saved: {history_path}")
        return history_path
    
    def save_training_curves(self, history: Dict, title: str, 
                            model_type: str = 'a') -> Path:
        """
        Sauvegarde les courbes d'entraînement (loss + accuracy).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=4)
        ax1.plot(epochs, history['val_loss'], 'r-o', label='Val', markersize=4)
        ax1.set_title('Loss (lower = better)', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=4)
        ax2.plot(epochs, history['val_acc'], 'r-o', label='Val', markersize=4)
        ax2.set_title('Accuracy (higher = better)', fontsize=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Déterminer le nom du fichier
        if self.disease == 'Alzheimer':
            plot_name = 'training_curves.png'
        else:
            plot_name = f'training_curves_{model_type}.png'
        
        plot_path = self.paths['plots'] / plot_name
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training curves saved: {plot_path}")
        return plot_path
    
    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], title: str,
                             model_type: str = 'a') -> Path:
        """
        Sauvegarde la matrice de confusion.
        """
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=class_names)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(title, fontweight='bold', fontsize=12)
        plt.tight_layout()
        
        # Déterminer le nom du fichier
        if self.disease == 'Alzheimer':
            cm_name = 'confusion_matrix.png'
        else:
            cm_name = f'confusion_matrix_{model_type}.png'
        
        cm_path = self.paths['confusion_matrix'] / cm_name
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved: {cm_path}")
        return cm_path
    
    def save_model_weights(self, model: torch.nn.Module, model_type: str = 'a') -> Path:
        """
        Sauvegarde les poids du modèle.
        """
        from src.config import get_model_save_path
        
        weight_path = get_model_save_path(self.disease, model_type)
        torch.save(model.state_dict(), weight_path)
        
        print(f"✅ Model weights saved: {weight_path}")
        return weight_path
    
    def save_metrics_report(self, metrics: Dict[str, Any], 
                           model_type: str = 'a') -> Path:
        """
        Sauvegarde un rapport des métriques d'évaluation.
        """
        if self.disease == 'Alzheimer':
            report_path = self.paths['history'] / 'metrics_report.json'
        else:
            report_path = self.paths['history'] / f'metrics_report_{model_type}.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics report saved: {report_path}")
        return report_path
    
    def print_structure(self) -> None:
        """Affiche la structure de sauvegarde créée."""
        print(f"\n{'='*70}")
        print(f"  📁 Output Structure for {self.disease}")
        print(f"{'='*70}")
        print(f"  Disease Root : {self.paths['disease_root']}")
        print(f"  ├── weights/           : Model weights (.pth)")
        print(f"  ├── history/           : Training history & metrics (.json)")
        print(f"  ├── plots/             : Training curves (.png)")
        print(f"  └── confusion_matrix/  : Confusion matrices (.png)")
        print(f"{'='*70}\n")


def get_output_manager(disease: str) -> OutputManager:
    """Factory pour obtenir un gestionnaire d'outputs."""
    return OutputManager(disease)
