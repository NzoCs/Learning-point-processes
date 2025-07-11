#!/usr/bin/env python3
"""
Guide de Démarrage Rapide - EasyTPP CLI

Ce script interactif guide l'utilisateur dans la configuration
et l'utilisation initiale de l'outil EasyTPP CLI.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Affiche l'en-tête du guide"""
    print("=" * 60)
    print("    🚀 GUIDE DE DÉMARRAGE RAPIDE - EasyTPP CLI")
    print("=" * 60)
    print()

def print_step(step_num, title):
    """Affiche un titre d'étape"""
    print(f"\n📋 ÉTAPE {step_num}: {title}")
    print("-" * 40)

def run_command(command, description=""):
    """Exécute une commande avec description"""
    if description:
        print(f"🔧 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print("   ✅ Succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erreur: {e}")
        return False

def check_python():
    """Vérifie la version de Python"""
    print("🐍 Vérification de Python...")
    try:
        import sys
        version = sys.version_info
        print(f"   Version Python: {version.major}.{version.minor}.{version.micro}")
        
        if version.major >= 3 and version.minor >= 8:
            print("   ✅ Version Python compatible")
            return True
        else:
            print("   ❌ Python 3.8+ requis")
            return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def check_files():
    """Vérifie que les fichiers nécessaires existent"""
    print("📁 Vérification des fichiers...")
    
    required_files = [
        "easytpp_cli.py",
        "setup_cli.py",
        "requirements-cli.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} manquant")
            missing_files.append(file)
    
    return len(missing_files) == 0

def installation_guide():
    """Guide d'installation"""
    print_step(1, "INSTALLATION")
    
    print("Choisissez votre méthode d'installation:")
    print("1. Installation automatique (recommandée)")
    print("2. Installation manuelle")
    print("3. Passer (déjà installé)")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        print("\n🔄 Installation automatique...")
        success = run_command("python setup_cli.py", "Exécution du script d'installation")
        if not success:
            print("⚠️ L'installation automatique a échoué. Essayez l'installation manuelle.")
            return False
            
    elif choice == "2":
        print("\n🔧 Installation manuelle...")
        print("Exécutez les commandes suivantes:")
        print("1. pip install -r requirements.txt")
        print("2. pip install -r requirements-cli.txt")
        print("3. python setup_cli.py")
        
        input("\nAppuyez sur Entrée quand vous avez terminé...")
        
    elif choice == "3":
        print("✅ Installation ignorée")
    else:
        print("❌ Choix invalide")
        return False
    
    return True

def basic_test():
    """Test de base du CLI"""
    print_step(2, "TEST DE BASE")
    
    print("🧪 Test des commandes de base...")
    
    tests = [
        ("python easytpp_cli.py --version", "Test de version"),
        ("python easytpp_cli.py --help", "Test d'aide"),
        ("python easytpp_cli.py info", "Test d'informations système")
    ]
    
    all_passed = True
    for command, description in tests:
        success = run_command(command, description)
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ Tous les tests de base sont passés!")
    else:
        print("\n❌ Certains tests ont échoué. Vérifiez l'installation.")
    
    return all_passed

def configuration_guide():
    """Guide de configuration"""
    print_step(3, "CONFIGURATION")
    
    print("📁 Vérification des répertoires...")
    directories = ["configs", "outputs", "logs", "checkpoints"]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"   ✅ {directory}/")
        else:
            print(f"   📁 Création de {directory}/")
            Path(directory).mkdir(exist_ok=True)
    
    print("\n⚙️ Fichiers de configuration disponibles:")
    configs_dir = Path("configs")
    if configs_dir.exists():
        config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
        if config_files:
            for config in config_files:
                print(f"   📄 {config.name}")
        else:
            print("   ⚠️ Aucun fichier de configuration trouvé")
            print("   💡 Créez un fichier de configuration ou utilisez l'exemple fourni")
    
    return True

def first_run_guide():
    """Guide de première exécution"""
    print_step(4, "PREMIÈRE EXÉCUTION")
    
    print("🎯 Options pour votre première exécution:")
    print("1. Mode interactif (recommandé pour débuter)")
    print("2. Lister les configurations disponibles")
    print("3. Valider une configuration")
    print("4. Voir les informations système")
    print("5. Passer")
    
    choice = input("\nVotre choix (1-5): ").strip()
    
    if choice == "1":
        print("\n🎯 Lancement du mode interactif...")
        run_command("python easytpp_cli.py interactive", "Mode interactif")
        
    elif choice == "2":
        print("\n📋 Configurations disponibles:")
        run_command("python easytpp_cli.py list-configs --dir configs", "Liste des configurations")
        
    elif choice == "3":
        config_path = input("Chemin du fichier de configuration: ").strip()
        experiment = input("ID de l'expérience (ex: THP): ").strip() or "THP"
        dataset = input("ID du dataset (ex: H2expc): ").strip() or "H2expc"
        
        command = f'python easytpp_cli.py validate --config "{config_path}" --experiment {experiment} --dataset {dataset}'
        run_command(command, "Validation de configuration")
        
    elif choice == "4":
        print("\n💻 Informations système:")
        run_command("python easytpp_cli.py info", "Informations système")
        
    elif choice == "5":
        print("✅ Première exécution ignorée")
    else:
        print("❌ Choix invalide")
        return False
    
    return True

def show_next_steps():
    """Affiche les prochaines étapes"""
    print_step(5, "PROCHAINES ÉTAPES")
    
    print("🎉 Félicitations! EasyTPP CLI est maintenant configuré.")
    print("\n📚 Ressources utiles:")
    print("   📖 CLI_PROFESSIONAL_README.md - Documentation principale")
    print("   📖 CLI_README.md - Documentation détaillée")
    print("   📁 configs/ - Fichiers de configuration")
    print("   📁 examples/ - Scripts d'exemple")
    
    print("\n🚀 Commandes essentielles à retenir:")
    print("   python easytpp_cli.py --help                    # Aide générale")
    print("   python easytpp_cli.py interactive               # Mode interactif")
    print("   python easytpp_cli.py list-configs              # Voir les configs")
    print("   python easytpp_cli.py info                      # Infos système")
    print("   python easytpp_cli.py run --help                # Aide pour run")
    
    print("\n🎯 Exemple d'utilisation complète:")
    print("   python easytpp_cli.py run \\")
    print("     --config configs/example_full_config.yaml \\")
    print("     --experiment THP \\")
    print("     --dataset H2expc \\")
    print("     --phase test")
    
    print("\n💡 Conseils:")
    print("   - Commencez par le mode interactif")
    print("   - Validez toujours vos configurations avant l'exécution")
    print("   - Utilisez --verbose pour plus de détails")
    print("   - Consultez la documentation pour les fonctionnalités avancées")

def main():
    """Fonction principale du guide"""
    print_header()
    
    print("Ce guide vous aidera à configurer et utiliser EasyTPP CLI.")
    print("Le processus prend environ 5-10 minutes.")
    
    # Vérifications préliminaires
    if not check_python():
        print("\n❌ Version Python incompatible. Installez Python 3.8+")
        sys.exit(1)
    
    if not check_files():
        print("\n❌ Fichiers manquants. Vérifiez l'installation.")
        sys.exit(1)
    
    print("\n✅ Vérifications préliminaires réussies!")
    
    # Guide étape par étape
    try:
        if not installation_guide():
            print("\n❌ Échec de l'installation")
            sys.exit(1)
        
        if not basic_test():
            print("\n⚠️ Tests de base échoués, mais on continue...")
        
        if not configuration_guide():
            print("\n❌ Échec de la configuration")
            sys.exit(1)
        
        if not first_run_guide():
            print("\n⚠️ Première exécution échouée, mais on continue...")
        
        show_next_steps()
        
        print("\n" + "=" * 60)
        print("🎉 CONFIGURATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)
        print("\nEasyTPP CLI est maintenant prêt à l'emploi.")
        print("Consultez CLI_PROFESSIONAL_README.md pour plus d'informations.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Guide interrompu par l'utilisateur.")
        print("Vous pouvez relancer ce guide à tout moment avec: python quick_start.py")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("Consultez la documentation ou contactez le support.")
        sys.exit(1)

if __name__ == "__main__":
    main()
