"""
init_structure.py
──────────────────
Script simple pour initialiser la structure des dossiers.
À exécuter une seule fois après le setup.
"""

import sys
from pathlib import Path

# Ajouter le projet root au path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    from src.config import config_manager
    
    print("\n" + "="*70)
    print("  Initialisation de la structure du projet")
    print("="*70 + "\n")
    
    diseases = ['Brain_Tumor', 'Alzheimer', 'Stroke']
    
    for disease in diseases:
        print(f"📁 Création de la structure pour {disease}...")
        
        # Créer la structure
        paths = config_manager.get_paths(disease)
        
        # Afficher
        config_manager.print_structure(disease)
        print(f"✅ {disease} - Structure créée avec succès!\n")
    
    print("="*70)
    print("✅ INITIALISATION TERMINÉE")
    print("="*70)
    print("\n📖 Prochaines étapes:")
    print("   1. Consultez STRUCTURE.md pour comprendre la nouvelle structure")
    print("   2. Consultez EXAMPLE_MIGRATION.md pour adapter vos notebooks")
    print("   3. Exécutez vos notebooks (train.ipynb, train_alzheimer.ipynb, etc.)")
    print("   4. Les outputs seront automatiquement sauvegardés au bon endroit\n")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
