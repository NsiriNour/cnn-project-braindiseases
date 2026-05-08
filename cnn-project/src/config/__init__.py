"""
config package
──────────────
Gestion centralisée de la configuration du projet.
"""

from .config_manager import (
    ConfigManager,
    config_manager,
    get_config,
    get_paths,
    get_model_save_path
)

__all__ = [
    'ConfigManager',
    'config_manager',
    'get_config',
    'get_paths',
    'get_model_save_path'
]
