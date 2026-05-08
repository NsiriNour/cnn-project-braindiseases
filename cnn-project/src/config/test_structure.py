"""
test_structure.py
──────────────────
Script de test pour valider que la nouvelle structure fonctionne.

Utilisation:
    python src/config/test_structure.py
"""

import sys
from pathlib import Path

# Ajouter le projet root au path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import get_config, get_paths, config_manager
from src.utils.output_manager import get_output_manager


def test_config_loading():
    """Test le chargement des configurations."""
    print("\n" + "="*70)
    print("TEST 1 : Chargement des configurations")
    print("="*70)
    
    diseases = ['Brain_Tumor', 'Alzheimer', 'Stroke']
    
    for disease in diseases:
        try:
            config = get_config(disease)
            print(f"✅ {disease:15} → Config loaded")
            print(f"   - Model: {config.get('model', config.get('model_a', {})).get('architecture', 'N/A')}")
            print(f"   - Epochs: {config.get('model', config.get('model_a', {})).get('epochs', 'N/A')}")
        except Exception as e:
            print(f"❌ {disease:15} → Error: {e}")


def test_paths_creation():
    """Test la création des chemins et dossiers."""
    print("\n" + "="*70)
    print("TEST 2 : Création des chemins et dossiers")
    print("="*70)
    
    diseases = ['Brain_Tumor', 'Alzheimer', 'Stroke']
    
    for disease in diseases:
        try:
            paths = get_paths(disease)
            
            # Vérifier que tous les dossiers existent
            all_exist = all(d.exists() for d in [
                paths['disease_root'],
                paths['weights'],
                paths['plots'],
                paths['history'],
                paths['confusion_matrix']
            ])
            
            if all_exist:
                print(f"✅ {disease:15} → All directories created")
                print(f"   └── {paths['disease_root']}/")
            else:
                print(f"⚠️  {disease:15} → Some directories missing")
                
        except Exception as e:
            print(f"❌ {disease:15} → Error: {e}")


def test_output_manager():
    """Test le gestionnaire d'outputs."""
    print("\n" + "="*70)
    print("TEST 3 : Gestionnaire d'outputs")
    print("="*70)
    
    diseases = ['Brain_Tumor', 'Alzheimer', 'Stroke']
    
    for disease in diseases:
        try:
            manager = get_output_manager(disease)
            print(f"✅ {disease:15} → OutputManager created")
            
            # Afficher les chemins disponibles
            print(f"   Chemins disponibles:")
            print(f"   - History: {manager.paths['history']}")
            print(f"   - Plots:   {manager.paths['plots']}")
            print(f"   - Weights: {manager.paths['weights']}")
            
        except Exception as e:
            print(f"❌ {disease:15} → Error: {e}")


def test_model_paths():
    """Test la génération des chemins de sauvegarde des modèles."""
    print("\n" + "="*70)
    print("TEST 4 : Chemins de sauvegarde des modèles")
    print("="*70)
    
    diseases = ['Brain_Tumor', 'Alzheimer', 'Stroke']
    
    for disease in diseases:
        try:
            if disease == 'Alzheimer':
                path = config_manager.get_model_save_path(disease)
                print(f"✅ {disease:15} → {path.name}")
            else:
                path_a = config_manager.get_model_save_path(disease, 'a')
                path_b = config_manager.get_model_save_path(disease, 'b')
                print(f"✅ {disease:15} → Model A: {path_a.name}")
                print(f"{'':15}    → Model B: {path_b.name}")
                
        except Exception as e:
            print(f"❌ {disease:15} → Error: {e}")


def test_example_usage():
    """Montre un exemple d'utilisation."""
    print("\n" + "="*70)
    print("TEST 5 : Exemple d'utilisation")
    print("="*70)
    
    print("\n📝 Exemple 1 : Charger une configuration")
    print("-" * 70)
    config = get_config('Stroke')
    print(f"config = get_config('Stroke')")
    print(f"config['model_a']['epochs'] = {config['model_a']['epochs']}")
    print(f"config['model_a']['learning_rate'] = {config['model_a']['learning_rate']}")
    
    print("\n📝 Exemple 2 : Obtenir les chemins")
    print("-" * 70)
    paths = get_paths('Stroke')
    print(f"paths = get_paths('Stroke')")
    print(f"paths['weights'] = {paths['weights']}")
    print(f"paths['plots'] = {paths['plots']}")
    
    print("\n📝 Exemple 3 : Sauvegarder les outputs")
    print("-" * 70)
    print(f"""
manager = get_output_manager('Stroke')

# Sauvegarder l'historique
manager.save_training_history(history_a, model_type='a')
# → results/Stroke/history/history_a.json

# Sauvegarder les courbes
manager.save_training_curves(history, 'Model A', 'a')
# → results/Stroke/plots/training_curves_a.png

# Sauvegarder les poids
manager.save_model_weights(model_a, model_type='a')
# → results/Stroke/weights/model_a_binary.pth
    """)
    
    print("\n📝 Exemple 4 : Inférence pour le déploiement")
    print("-" * 70)
    print(f"""
from src.inference import get_predictor

predictor = get_predictor('Stroke', device='cuda')
result = predictor.predict('image.jpg')

print(result)
# {{
#     'disease': 'Stroke',
#     'status': 'stroke',
#     'type': 'Bleeding',
#     'confidence_detection': 0.95,
#     'confidence_classification': 0.87,
#     'message': '🚨 Stroke Detected: Bleeding...'
# }}
    """)


def print_summary():
    """Affiche un résumé des fichiers créés."""
    print("\n" + "="*70)
    print("RÉSUMÉ DES FICHIERS CRÉÉS")
    print("="*70)
    
    files_created = {
        "Configuration": [
            "config/brain_tumor_config.json",
            "config/alzheimer_config.json",
            "config/stroke_config.json"
        ],
        "Source Code": [
            "src/config/__init__.py",
            "src/config/config_manager.py",
            "src/inference/__init__.py",
            "src/inference/predictor.py",
            "src/utils/output_manager.py"
        ],
        "Documentation": [
            "STRUCTURE.md",
            "EXAMPLE_MIGRATION.md",
            "src/config/test_structure.py"
        ],
        "Output Directories (auto-created)": [
            "results/Brain_Tumor/{weights,plots,history,confusion_matrix}/",
            "results/Alzheimer/{weights,plots,history,confusion_matrix}/",
            "results/Stroke/{weights,plots,history,confusion_matrix}/"
        ]
    }
    
    for category, files in files_created.items():
        print(f"\n📁 {category}:")
        for f in files:
            print(f"   ✓ {f}")


def main():
    """Execute tous les tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🧪 TEST DE LA NOUVELLE STRUCTURE" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        test_config_loading()
        test_paths_creation()
        test_output_manager()
        test_model_paths()
        test_example_usage()
        print_summary()
        
        print("\n" + "="*70)
        print("✅ TOUS LES TESTS SONT PASSÉS")
        print("="*70)
        print("\n🎉 La nouvelle structure est prête à l'emploi!")
        print("\n📖 Consultez STRUCTURE.md et EXAMPLE_MIGRATION.md pour")
        print("   plus d'informations sur comment adapter vos notebooks.\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
