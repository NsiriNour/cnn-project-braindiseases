"""
config_manager.py
─────────────────
Gestionnaire centralisé de configuration pour tous les modèles.
Charge les configurations JSON et crée automatiquement la structure de dossiers.
"""

import json
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """
    Gère les configurations de chaque maladie.
    Charge automatiquement les fichiers JSON et crée la structure de dossiers.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]  # cnn-project/
        self.config_dir = self.project_root / 'config'
        self.results_dir = self.project_root / 'results'
        
        # Créer le dossier results s'il n'existe pas
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs = {}
    
    def load_config(self, disease: str) -> Dict[str, Any]:
        """
        Charge la configuration pour une maladie donnée.
        
        Args:
            disease: 'Brain_Tumor', 'Alzheimer', ou 'Stroke'
        
        Returns:
            Configuration dict
        """
        if disease not in self.configs:
            config_file = self._get_config_file(disease)
            with open(config_file, 'r', encoding='utf-8') as f:
                self.configs[disease] = json.load(f)
        
        return self.configs[disease]
    
    def _get_config_file(self, disease: str) -> Path:
        """Retourne le chemin du fichier de configuration."""
        disease_map = {
            'Brain_Tumor': 'brain_tumor_config.json',
            'Alzheimer': 'alzheimer_config.json',
            'Stroke': 'stroke_config.json'
        }
        
        if disease not in disease_map:
            raise ValueError(f"Disease '{disease}' not supported. "
                           f"Choose from: {list(disease_map.keys())}")
        
        config_file = self.config_dir / disease_map[disease]
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        return config_file
    
    def create_disease_structure(self, disease: str) -> Dict[str, Path]:
        """
        Crée la structure de dossiers pour une maladie.
        
        Returns:
            Dict avec tous les chemins (weights, plots, history, etc.)
        """
        disease_dir = self.results_dir / disease
        
        # Créer tous les sous-dossiers
        weights_dir = disease_dir / 'weights'
        plots_dir = disease_dir / 'plots'
        history_dir = disease_dir / 'history'
        confusion_dir = disease_dir / 'confusion_matrix'
        
        for d in [weights_dir, plots_dir, history_dir, confusion_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        return {
            'disease_root': disease_dir,
            'weights': weights_dir,
            'plots': plots_dir,
            'history': history_dir,
            'confusion_matrix': confusion_dir
        }
    
    def get_paths(self, disease: str) -> Dict[str, Path]:
        """
        Retourne tous les chemins structurés pour une maladie.
        Crée les dossiers s'ils n'existent pas.
        """
        paths = self.create_disease_structure(disease)
        paths['config'] = self._get_config_file(disease)
        return paths
    
    def get_model_save_path(self, disease: str, model_type: str = 'a') -> Path:
        """
        Retourne le chemin de sauvegarde pour un modèle.
        
        Args:
            disease: 'Brain_Tumor', 'Alzheimer', ou 'Stroke'
            model_type: 'a' ou 'b' (pour Brain_Tumor et Stroke)
        
        Example:
            results/Stroke/weights/model_a_binary.pth
        """
        config = self.load_config(disease)
        paths = config.get('paths', {})

        if disease == 'Alzheimer':
            path_str = paths.get('weights', None)
            if path_str is None:
                path_str = str(self.results_dir / disease / 'weights' / 'model_efficientnet_b0.pth')
        elif disease == 'Brain_Tumor':
            if model_type == 'a':
                path_str = paths.get('weights_a', None)
                if path_str is None:
                    path_str = str(self.results_dir / disease / 'weights' / 'model_a_binary.pth')
            else:
                path_str = paths.get('weights_b', None)
                if path_str is None:
                    path_str = str(self.results_dir / disease / 'weights' / 'model_b_3class.pth')
        elif disease == 'Stroke':
            if model_type == 'a':
                path_str = paths.get('weights_a', None)
                if path_str is None:
                    path_str = str(self.results_dir / disease / 'weights' / 'model_a_binary.pth')
            else:
                path_str = paths.get('weights_b', None)
                if path_str is None:
                    path_str = str(self.results_dir / disease / 'weights' / 'model_b_multiclass.pth')
        else:
            raise ValueError(f"Unknown disease: {disease}")

        model_path = Path(path_str)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        return model_path
    
    def get_history_save_path(self, disease: str, model_type: str = 'a') -> Path:
        """
        Retourne le chemin de sauvegarde pour l'historique d'entraînement.
        
        Example:
            results/Stroke/history/history_a.json
        """
        config = self.load_config(disease)
        paths = config.get('paths', {})

        if disease == 'Alzheimer':
            path_str = paths.get('history', None)
            if path_str is None:
                path_str = str(self.results_dir / disease / 'history' / 'history.json')
        else:
            if model_type == 'a':
                path_str = paths.get('history_a', None)
            else:
                path_str = paths.get('history_b', None)
            if path_str is None:
                path_str = str(self.results_dir / disease / 'history' / f'history_{model_type}.json')

        history_path = Path(path_str)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        return history_path
    
    def get_plot_save_path(self, disease: str, plot_name: str) -> Path:
        """
        Retourne le chemin de sauvegarde pour un graphique.
        
        Example:
            results/Stroke/plots/training_curves_a.png
        """
        config = self.load_config(disease)
        paths = config.get('paths', {})
        plots_dir = paths.get('plots', None)

        if plots_dir is None:
            plots_path = self.results_dir / disease / 'plots' / f"{plot_name}.png"
        else:
            plots_path = Path(plots_dir) / f"{plot_name}.png"

        plots_path.parent.mkdir(parents=True, exist_ok=True)
        return plots_path
    
    def get_confusion_matrix_path(self, disease: str, model_type: str = 'a') -> Path:
        """
        Retourne le chemin de sauvegarde pour une matrice de confusion.
        
        Example:
            results/Stroke/confusion_matrix/confusion_matrix_a.png
        """
        config = self.load_config(disease)
        paths = config.get('paths', {})

        if disease == 'Alzheimer':
            path_str = paths.get('confusion_matrix', None)
            if path_str is None:
                path_str = str(self.results_dir / disease / 'confusion_matrix' / 'confusion_matrix.png')
        else:
            if model_type == 'a':
                path_str = paths.get('confusion_matrix_a', None)
            else:
                path_str = paths.get('confusion_matrix_b', None)
            if path_str is None:
                path_str = str(self.results_dir / disease / 'confusion_matrix' / f'confusion_matrix_{model_type}.png')

        cm_path = Path(path_str)
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        return cm_path
    
    def print_structure(self, disease: str) -> None:
        """Affiche la structure de dossiers créée."""
        paths = self.get_paths(disease)
        print(f"\n{'='*60}")
        print(f"  Structure pour {disease}")
        print(f"{'='*60}")
        for key, path in paths.items():
            print(f"  {key:20} : {path}")
        print(f"{'='*60}\n")


# ────────────────────────────────────────────────────────────
# Instance globale pour utilisation simplifiée
# ────────────────────────────────────────────────────────────

config_manager = ConfigManager()


def get_config(disease: str) -> Dict[str, Any]:
    """Raccourci pour charger une configuration."""
    return config_manager.load_config(disease)


def get_paths(disease: str) -> Dict[str, Path]:
    """Raccourci pour obtenir tous les chemins d'une maladie."""
    return config_manager.get_paths(disease)


def get_model_save_path(disease: str, model_type: str = 'a') -> Path:
    """Raccourci pour obtenir le chemin de sauvegarde d'un modèle."""
    return config_manager.get_model_save_path(disease, model_type)
