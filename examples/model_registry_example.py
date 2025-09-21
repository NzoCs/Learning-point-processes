"""
Exemple d'utilisation du ModelRegistry

Ce script montre comment utiliser le registry pour vérifier que l'enum Models
est à jour avec tous les modèles disponibles.
"""

from easy_tpp.models.model_registry import ModelRegistry


def example_basic_validation():
    """Exemple de validation basique."""
    print("=== Validation basique ===")

    registry = ModelRegistry()
    is_valid = registry.validate_models()

    print(f"L'enum est-il à jour ? {is_valid}")


def example_detailed_report():
    """Exemple de rapport détaillé."""
    print("\n=== Rapport détaillé ===")

    registry = ModelRegistry()
    registry.print_validation_report()


def example_check_missing_models():
    """Exemple pour vérifier les modèles manquants."""
    print("\n=== Vérification des modèles manquants ===")

    registry = ModelRegistry()

    missing = registry.get_missing_models()
    if missing:
        print(f"Modèles manquants: {missing}")
        print("Vous devriez les ajouter à l'enum Models!")
    else:
        print("Aucun modèle manquant ✅")

    extra = registry.get_extra_models()
    if extra:
        print(f"Modèles en trop: {extra}")
        print("Ces modèles sont dans l'enum mais pas trouvés dans le code")
    else:
        print("Aucun modèle en trop ✅")


def example_generate_enum_code():
    """Exemple pour générer le code de l'enum."""
    print("\n=== Génération du code enum ===")

    registry = ModelRegistry()
    code = registry.generate_enum_code()

    print("Code suggéré pour l'enum Models:")
    print("-" * 40)
    print(code)


def example_get_all_models():
    """Exemple pour obtenir tous les modèles."""
    print("\n=== Liste de tous les modèles ===")

    registry = ModelRegistry()

    discovered = registry.get_all_discovered_models()
    print(f"Modèles découverts ({len(discovered)}):")
    for name, model_class in discovered.items():
        module = model_class.__module__.split(".")[-1]
        print(f"  - {name} (dans {module}.py)")

    enum_models = registry.get_enum_models()
    print(f"\nModèles dans l'enum ({len(enum_models)}):")
    for name in enum_models.keys():
        print(f"  - {name}")


def main():
    """Fonction principale pour tous les exemples."""
    print("🔍 Exemples ModelRegistry")
    print("=" * 50)

    try:
        example_basic_validation()
        example_detailed_report()
        example_check_missing_models()
        example_get_all_models()

        # Générer le code seulement si nécessaire
        registry = ModelRegistry()
        if not registry.validate_models():
            example_generate_enum_code()

    except Exception as e:
        print(f"Erreur dans les exemples: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("✅ Exemples terminés!")


def quick_check():
    """Fonction rapide pour un check complet."""
    print("🚀 Vérification rapide des modèles")
    print("=" * 40)

    validate_models_registry()


if __name__ == "__main__":
    # Utiliser quick_check() pour une vérification rapide
    # ou main() pour tous les exemples détaillés
    quick_check()
    # main()
