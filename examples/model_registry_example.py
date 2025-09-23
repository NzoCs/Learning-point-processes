"""
Exemple d'utilisation du ModelRegistry

Ce script montre comment utiliser le registry automatique pour accéder aux modèles TPP
sans avoir besoin de les enregistrer manuellement.
"""

from easy_tpp.models.model_registry import ModelRegistry
from easy_tpp.models import *  # Import tous les modèles pour qu'ils s'enregistrent


def example_list_all_models():
    """Exemple pour lister tous les modèles enregistrés."""
    print("=== Liste de tous les modèles enregistrés ===")
    
    models = ModelRegistry.list_models()
    print(f"Nombre de modèles enregistrés: {len(models)}")
    
    for model_name in sorted(models):
        model_class = ModelRegistry.get_model(model_name)
        module_name = model_class.__module__.split('.')[-1]
        print(f"  - {model_name} (dans {module_name}.py)")


def example_get_specific_model():
    """Exemple pour récupérer un modèle spécifique."""
    print("\n=== Récupération d'un modèle spécifique ===")
    
    # Vérifier si un modèle existe
    model_name = "NHP"
    if ModelRegistry.model_exists(model_name):
        model_class = ModelRegistry.get_model(model_name)
        print(f"✅ Modèle '{model_name}' trouvé: {model_class}")
        print(f"   Module: {model_class.__module__}")
        print(f"   Classe: {model_class.__name__}")
    else:
        print(f"❌ Modèle '{model_name}' non trouvé")
    
    # Tester avec un modèle inexistant
    fake_model = "FAKE_MODEL"
    if ModelRegistry.model_exists(fake_model):
        print(f"✅ Modèle '{fake_model}' trouvé")
    else:
        print(f"❌ Modèle '{fake_model}' non trouvé (normal)")


def example_get_registry():
    """Exemple pour obtenir le registry complet."""
    print("\n=== Registry complet ===")
    
    registry = ModelRegistry.get_registry()
    print(f"Registry contient {len(registry)} modèles:")
    
    for name, model_class in registry.items():
        print(f"  {name}: {model_class.__name__}")


def example_model_types():
    """Exemple pour analyser les types de modèles."""
    print("\n=== Analyse des types de modèles ===")
    
    models = ModelRegistry.get_registry()
    
    # Grouper par module
    by_module = {}
    for name, model_class in models.items():
        module = model_class.__module__.split('.')[-1]
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(name)
    
    for module, model_names in by_module.items():
        print(f"  {module}.py: {', '.join(sorted(model_names))}")


if __name__ == "__main__":
    print("🔄 Démarrage de l'exemple ModelRegistry")
    
    example_list_all_models()
    example_get_specific_model()
    example_get_registry()
    example_model_types()
    
    print("\n✅ Exemple terminé avec succès!")
