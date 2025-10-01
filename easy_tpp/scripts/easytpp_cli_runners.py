#!/usr/bin/env python3
"""
EasyTPP CLI v4.0 - Processus Runners Architecture

Interface en ligne de commande pour EasyTPP utilisant des runners modulaires
pour chaque processus (expérience, inspection, génération, etc.).

Usage:
    python easytpp_cli_runners.py --help
    python easytpp_cli_runners.py run --config config.yaml
    python easytpp_cli_runners.py inspect --data-dir ./data/
"""

import sys
from pathlib import Path
from typing import Optional, List

try:
    import typer
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    typer = None
    Console = None
    rprint = print

# Import des runners
try:
    from easy_tpp.runners import (
        ExperimentRunner,
        DataInspector,
        SystemInfo,
        InteractiveSetup,
        BenchmarkRunner,
        DataGenerator
    )
    RUNNERS_AVAILABLE = True
except ImportError as e:
    RUNNERS_AVAILABLE = False
    IMPORT_ERROR = str(e)

def check_requirements():
    """Vérifie que les dépendances sont disponibles."""
    if not TYPER_AVAILABLE:
        print("❌ Erreur: typer n'est pas installé")
        print("Installation: pip install typer[all] rich")
        return False
        
    if not RUNNERS_AVAILABLE:
        print(f"❌ Erreur: Runners non disponibles - {IMPORT_ERROR}")
        print("Vérifiez l'installation d'EasyTPP")
        return False
        
    return True

# Configuration de l'application - après la vérification
if not check_requirements():
    sys.exit(1)

app = typer.Typer(
    name="easytpp",
    help="EasyTPP CLI v4.0 - Temporal Point Processes avec architecture runners",
    no_args_is_help=True
)
console = Console()

@app.command("run")
def run_experiment(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Fichier de configuration"),
    dataset_id: Optional[str] = typer.Option(None, "--dataset", "-d", help="ID du dataset"),
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="ID du modèle"),
    phase: str = typer.Option("all", "--phase", "-p", help="Phase d'exécution (train/test/predict/all)"),
    max_epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Nombre maximum d'époques"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", "-s", help="Répertoire de sauvegarde"),
    gpu_id: Optional[int] = typer.Option(None, "--gpu", "-g", help="ID du GPU à utiliser"),
    debug: bool = typer.Option(False, "--debug", help="Mode debug")
):
    """Lance une expérience TPP avec ExperimentRunner."""
    runner = ExperimentRunner()
    success = runner.run_experiment(
        config_path=config,
        dataset_id=dataset_id,
        model_id=model_id,
        phase=phase,
        max_epochs=max_epochs,
        save_dir=save_dir,
        gpu_id=gpu_id,
        debug=debug
    )
    
    if not success:
        raise typer.Exit(1)

@app.command("inspect")
def inspect_data(
    data_dir: str = typer.Argument(..., help="Répertoire contenant les données"),
    data_format: str = typer.Option("json", "--format", "-f", help="Format des données"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Répertoire de sortie"),
    save_graphs: bool = typer.Option(True, "--save/--no-save", help="Sauvegarder les graphiques"),
    show_graphs: bool = typer.Option(False, "--show/--no-show", help="Afficher les graphiques"),
    max_sequences: Optional[int] = typer.Option(None, "--max-seq", help="Nombre max de séquences")
):
    """Inspecte et visualise des données TPP avec DataInspector."""
    runner = DataInspector()
    success = runner.inspect_data(
        data_dir=data_dir,
        data_format=data_format,
        output_dir=output_dir,
        save_graphs=save_graphs,
        show_graphs=show_graphs,
        max_sequences=max_sequences
    )
    
    if not success:
        raise typer.Exit(1)

@app.command("generate")
def generate_data(
    output_dir: str = typer.Argument(..., help="Répertoire de sortie"),
    num_sequences: int = typer.Option(1000, "--num-seq", "-n", help="Nombre de séquences"),
    max_seq_len: int = typer.Option(100, "--max-len", "-l", help="Longueur max des séquences"),
    num_event_types: int = typer.Option(5, "--event-types", "-t", help="Nombre de types d'événements"),
    method: str = typer.Option("nhp", "--method", "-m", help="Méthode de génération"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Fichier de configuration"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Graine pour reproductibilité")
):
    """Génère des données synthétiques TPP avec DataGenerator."""
    runner = DataGenerator()
    success = runner.generate_data(
        output_dir=output_dir,
        num_sequences=num_sequences,
        max_seq_len=max_seq_len,
        num_event_types=num_event_types,
        generation_method=method,
        config_path=config,
        seed=seed
    )
    
    if not success:
        raise typer.Exit(1)

# ConfigValidator supprimé - utiliser la validation intégrée dans les configs

@app.command("info")
def system_info(
    include_deps: bool = typer.Option(True, "--deps/--no-deps", help="Inclure les dépendances"),
    include_hardware: bool = typer.Option(True, "--hw/--no-hw", help="Inclure le matériel"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Fichier de sortie")
):
    """Affiche les informations système avec SystemInfo."""
    runner = SystemInfo()
    success = runner.display_system_info(
        include_deps=include_deps,
        include_hardware=include_hardware,
        output_file=output
    )
    
    if not success:
        raise typer.Exit(1)

# ConfigManager supprimé - utiliser les outils de configuration intégrés

@app.command("setup")
def interactive_setup(
    setup_type: str = typer.Option("experiment", "--type", "-t", help="Type de setup"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Fichier de sortie"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Mode rapide")
):
    """Configuration interactive avec InteractiveSetup."""
    runner = InteractiveSetup()
    success = runner.run_interactive_setup(
        setup_type=setup_type,
        output_path=output,
        quick_mode=quick
    )
    
    if not success:
        raise typer.Exit(1)

@app.command("benchmark")
def benchmark_performance(
    configs: List[str] = typer.Argument(..., help="Fichiers de configuration à tester"),
    output_dir: str = typer.Option("./benchmarks", "--output", "-o", help="Répertoire de sortie"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Nombre d'itérations"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Mesurer la mémoire"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Mesurer le GPU")
):
    """Lance des benchmarks de performance avec BenchmarkRunner."""
    runner = BenchmarkRunner()
    success = runner.run_benchmark(
        configs=configs,
        output_dir=output_dir,
        iterations=iterations,
        include_memory=memory,
        include_gpu=gpu
    )
    
    if not success:
        raise typer.Exit(1)

@app.command("version")
def show_version():
    """Affiche la version du CLI."""
    console.print("[bold blue]EasyTPP CLI v4.0[/bold blue]")
    console.print("Architecture: Processus Runners")
    console.print("Runners disponibles:")
    
    runners = [
        "ExperimentRunner - Exécution d'expériences TPP",
        "DataInspector - Inspection et visualisation de données", 
        "DataGenerator - Génération de données synthétiques",
        "SystemInfo - Informations système",
        "InteractiveSetup - Configuration guidée",
        "BenchmarkRunner - Tests de performance"
    ]
    
    for runner in runners:
        console.print(f"  • [green]{runner}[/green]")

def main():
    """Point d'entrée principal."""
    if len(sys.argv) == 1:
        console.print("[bold blue]🚀 EasyTPP CLI v4.0 - Architecture Runners[/bold blue]")
        console.print()
        console.print("Commandes disponibles:")
        
        commands = [
            ("run", "Lancer une expérience TPP"),
            ("inspect", "Inspecter et visualiser des données"),
            ("generate", "Générer des données synthétiques"),
            ("info", "Informations système"),
            ("setup", "Configuration interactive"),
            ("benchmark", "Tests de performance"),
            ("version", "Afficher la version")
        ]
        
        table = Table()
        table.add_column("Commande", style="cyan")
        table.add_column("Description", style="white")
        
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        
        console.print(table)
        console.print()
        console.print("Utilisez [bold]--help[/bold] avec chaque commande pour plus de détails")
        return
    
    app()

if __name__ == "__main__":
    main()