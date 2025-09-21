"""
Exemple d'utilisation de la ModelFactory

Cet exemple montre comment utiliser la factory pour créer des modèles
au lieu d'utiliser BaseModel.generate_model_from_config directement.
"""

from easy_tpp.models.model_factory import Models, ModelFactory, create_model_from_config, quick_model
from easy_tpp.configs.model_config import ModelConfig


def example_create_model_with_factory():
    """Exemple de création de modèle avec la factory."""
    print("=== Création avec la factory ===")
    
    # Configuration du modèle (exemple)
    model_config = ModelConfig(
        model_id="NHP",
        # Ajoutez ici vos paramètres de configuration
    )
    
    # Méthode 1: Avec la factory directement
    factory = ModelFactory()
    model = factory.create_model(Models.NHP, model_config)
    print(f"Modèle créé: {model.__class__.__name__}")


def example_create_model_by_name():
    """Exemple de création par nom."""
    print("\n=== Création par nom ===")
    
    model_config = ModelConfig(model_id="SAHP")
    
    factory = ModelFactory()
    model = factory.create_model_by_name("SAHP", model_config)
    print(f"Modèle créé: {model.__class__.__name__}")


def example_compatibility_function():
    """Exemple avec la fonction de compatibilité."""
    print("\n=== Fonction de compatibilité ===")
    
    model_config = ModelConfig(model_id="THP")
    
    # Cette fonction reproduit le comportement de BaseModel.generate_model_from_config
    model = create_model_from_config(model_config)
    print(f"Modèle créé: {model.__class__.__name__}")


def example_quick_model():
    """Exemple avec la fonction rapide."""
    print("\n=== Fonction rapide ===")
    
    model_config = ModelConfig(model_id="RMTPP")
    
    model = quick_model(Models.RMTPP, model_config)
    print(f"Modèle créé: {model.__class__.__name__}")


def example_list_models():
    """Exemple pour lister les modèles disponibles."""
    print("\n=== Modèles disponibles ===")
    
    factory = ModelFactory()
    models = factory.list_available_models()
    
    print("Modèles découverts automatiquement:")
    for model_name in models:
        print(f"  - {model_name}")
    
    print("\nModèles disponibles dans l'enum:")
    for model in Models:
        print(f"  - {model.name}: {model.value}")


def example_error_handling():
    """Exemple de gestion d'erreur."""
    print("\n=== Gestion d'erreurs ===")
    
    factory = ModelFactory()
    
    # Essayer avec un modèle qui n'existe pas
    try:
        model_config = ModelConfig(model_id="NONEXISTENT")
        model = factory.create_model_by_name("NONEXISTENT", model_config)
    except ValueError as e:
        print(f"Erreur attendue: {e}")


def example_migration_from_basemodel():
    """Exemple de migration depuis BaseModel.generate_model_from_config."""
    print("\n=== Migration depuis BaseModel ===")
    
    model_config = ModelConfig(model_id="NHP")
    
    # AVANT: BaseModel.generate_model_from_config(model_config)
    # APRÈS: 
    model = create_model_from_config(model_config)
    print(f"Migration réussie: {model.__class__.__name__}")


def main():
    """Fonction principale pour tester tous les exemples."""
    print("🏭 Exemples ModelFactory")
    print("=" * 50)
    
    try:
        example_create_model_with_factory()
        example_create_model_by_name() 
        example_compatibility_function()
        example_quick_model()
        example_list_models()
        example_error_handling()
        example_migration_from_basemodel()
        
    except Exception as e:
        print(f"Erreur dans les exemples: {e}")
        print("Note: Certains exemples peuvent échouer si les modèles ne sont pas tous disponibles")
    
    print("\n" + "=" * 50)
    print("✅ Exemples terminés!")


if __name__ == "__main__":
    main()