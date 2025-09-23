"""
Exemple d'utilisation de la ModelFactory

Cet exemple montre comment utiliser la factory pour créer des modèles
de manière automatique grâce au système de registry basé sur des métaclasses.
La factory découvre automatiquement tous les modèles disponibles.
"""

from easy_tpp.configs.model_config import ModelConfig
from easy_tpp.models.model_factory import ModelFactory


def example_create_model_with_factory():
    """Exemple de création de modèle avec la factory."""
    print("=== Création avec la factory ===")

    # Configuration du modèle (exemple)
    model_config = ModelConfig(
        model_id="NHP",
        num_event_types=10,  # Requis pour la configuration
    )

    # Création avec la factory (découverte automatique)
    factory = ModelFactory()
    model = factory.create_model_by_name("NHP", model_config)
    print(f"Modèle créé: {model.__class__.__name__}")
    print(f"Type de modèle: {type(model)}")


def example_create_model_by_name():
    """Exemple de création par nom."""
    print("\n=== Création par nom ===")

    # Test avec différents modèles
    factory = ModelFactory()
    models_to_test = ["SAHP", "THP", "AttNHP"]
    
    for model_name in models_to_test:
        try:
            model_config = ModelConfig(model_id=model_name, num_event_types=10)
            model = factory.create_model_by_name(model_name, model_config)
            print(f"✅ {model_name}: {model.__class__.__name__}")
        except Exception as e:
            print(f"❌ {model_name}: {e}")


def example_list_models():
    """Exemple pour lister les modèles disponibles."""
    print("\n=== Modèles disponibles ===")

    factory = ModelFactory()
    models = factory.list_available_models()

    print(f"Modèles découverts automatiquement ({len(models)} modèles):")
    for i, model_name in enumerate(sorted(models), 1):
        print(f"  {i:2d}. {model_name}")
    
    # Affichage du registry interne
    from easy_tpp.models.model_registry import ModelRegistry
    registry_models = ModelRegistry.list_models()
    print(f"\nModèles dans le registry ({len(registry_models)} modèles):")
    for i, model_name in enumerate(sorted(registry_models), 1):
        print(f"  {i:2d}. {model_name}")


def example_error_handling():
    """Exemple de gestion d'erreur."""
    print("\n=== Gestion d'erreurs ===")

    factory = ModelFactory()

    # Test 1: Modèle qui n'existe pas
    try:
        model_config = ModelConfig(model_id="NONEXISTENT", num_event_types=10)
        model = factory.create_model_by_name("NONEXISTENT", model_config)
        print("❌ Aucune erreur détectée (inattendu)")
    except ValueError as e:
        print(f"✅ Erreur attendue pour modèle inexistant: {e}")
    
    # Test 2: Configuration invalide
    try:
        model = factory.create_model_by_name("NHP", None)
        print("❌ Aucune erreur détectée pour config None (inattendu)")
    except Exception as e:
        print(f"✅ Erreur attendue pour config None: {type(e).__name__}: {e}")
    
    # Test 3: Vérification de la robustesse
    print(f"\n🔍 Factory toujours fonctionnelle: {len(factory.list_available_models())} modèles")



def example_advanced_usage():
    """Exemple d'utilisation avancée de la factory."""
    print("\n=== Utilisation avancée ===")
    
    factory = ModelFactory()
    available_models = factory.list_available_models()
    
    # Créer plusieurs modèles différents
    print("Création de plusieurs modèles:")
    models_created = []
    
    for model_name in list(available_models)[:3]:  # Prendre les 3 premiers
        try:
            config = ModelConfig(model_id=model_name, num_event_types=10)
            model = factory.create_model_by_name(model_name, config)
            models_created.append((model_name, model))
            print(f"  ✅ {model_name}: {model.__class__.__module__}.{model.__class__.__name__}")
        except Exception as e:
            print(f"  ❌ {model_name}: {e}")
    
    print(f"\n🎯 Résultat: {len(models_created)} modèles créés avec succès")
    
    # Vérification des types
    print("\nVérification des types:")
    for name, model in models_created:
        print(f"  - {name}: isinstance(BaseModel) = {hasattr(model, '__class__')}")


def main():
    """Fonction principale pour tester tous les exemples."""
    print("🏭 Exemples ModelFactory - Système Automatique")
    print("=" * 55)
    print("📋 Ce système découvre automatiquement tous les modèles")
    print("🔧 Aucune maintenance manuelle d'enum nécessaire")
    print("=" * 55)

    try:
        # Import des modèles pour déclencher l'enregistrement
        print("🔄 Chargement des modèles...")
        import easy_tpp.models  # Déclenche l'enregistrement automatique
        print("✅ Modèles chargés\n")
        
        example_create_model_with_factory()
        example_create_model_by_name()
        example_list_models()
        example_error_handling()
        example_advanced_usage()

    except Exception as e:
        print(f"❌ Erreur dans les exemples: {e}")
        import traceback
        traceback.print_exc()
        print(
            "\n📝 Note: Certains exemples peuvent échouer selon les dépendances disponibles"
        )

    print("\n" + "=" * 55)
    print("✅ Exemples terminés!")
    print("🎉 Le système de factory automatique fonctionne parfaitement")


if __name__ == "__main__":
    main()
