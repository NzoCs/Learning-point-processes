#!/usr/bin/env python3
"""
Outil en ligne de commande pour exécuter les expériences EasyTPP

Ce script permet d'exécuter des expériences de Temporal Point Process
avec différents modèles et datasets configurés dans runner_config.yaml.

Usage:
    python run_model.py --help
    python run_model.py --list-experiments
    python run_model.py --list-datasets
    python run_model.py --experiment THP --dataset H2expc --phase train
    python run_model.py --experiment THP --dataset H2expc --phase test --checkpoint path/to/model.ckpt
    python run_model.py --all-experiments --dataset H2expc --phase test
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils.yaml_config_utils import parse_runner_yaml_config
from easy_tpp.utils import logger


@dataclass
class ExperimentInfo:
    """Information about an experiment."""
    name: str
    model_name: str
    description: str


class ModelRunner:
    """Runner for executing TPP model experiments."""
    
    def __init__(self, config_path: str = None, output_dir: str = None, debug: bool = False):
        """
        Initialize the model runner.
        
        Args:
            config_path: Path to runner_config.yaml file
            output_dir: Output directory for results
            debug: Enable debug mode for detailed error information
        """
        if config_path is None:
            config_path = Path(__file__).parent / "runner_config.yaml"
        
        self.config_path = Path(config_path)
        self.output_dir = output_dir or "./experiment_results"
        self.debug = debug
        
        # Load configuration to get available experiments and datasets
        self.config = self._load_config()
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load runner configuration to extract available experiments and datasets."""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def list_experiments(self) -> List[str]:
        """List all available experiments."""
        experiments = []
        if 'experiments' in self.config:
            experiments = list(self.config['experiments'].keys())
        else:
            # For the current config format, experiments are top-level sections
            # that contain model_config
            for key, value in self.config.items():
                if isinstance(value, dict) and 'model_config' in value:
                    experiments.append(key)
        return experiments
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        datasets = []
        if 'datasets' in self.config:
            datasets = list(self.config['datasets'].keys())
        elif 'data' in self.config:
            datasets = list(self.config['data'].keys())
        return datasets
    
    def run_experiment(self, experiment_id: str, dataset_id: str, phase: str, 
                      checkpoint_path: str = None) -> bool:
        """
        Run a single experiment.
        
        Args:
            experiment_id: ID of the experiment to run
            dataset_id: ID of the dataset to use
            phase: Phase to execute (train, test, predict, validation, all)
            checkpoint_path: Path to checkpoint file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting experiment: {experiment_id}")
            logger.info(f"📊 Dataset: {dataset_id}")
            logger.info(f"⚙️  Phase: {phase}")
            if checkpoint_path:
                logger.info(f"📂 Checkpoint: {checkpoint_path}")
            
            # Build configuration from YAML using the utility
            config_dict = parse_runner_yaml_config(
                str(self.config_path), 
                experiment_id, 
                dataset_id
            )
            
            config = RunnerConfig.from_dict(config_dict)
            
            # Create output directory for this experiment
            exp_output_dir = os.path.join(self.output_dir, f"{experiment_id}_{dataset_id}_{phase}")
            os.makedirs(exp_output_dir, exist_ok=True)
            
            runner = Runner(
                config=config,
                checkpoint_path=checkpoint_path,
                output_dir=exp_output_dir
            )
            
            runner.run(phase=phase)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Experiment completed successfully!")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"📁 Results saved to: {exp_output_dir}")
            
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Experiment failed: {str(e)}")
            logger.error(f"⏱️  Failed after: {execution_time:.2f} seconds")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            return False
    
    def run_all_experiments_on_dataset(self, dataset_id: str, phase: str) -> Dict[str, bool]:
        """Run all experiments on a single dataset."""
        experiments = self.list_experiments()
        results = {}
        
        logger.info(f"🔄 Running all experiments on dataset: {dataset_id}")
        logger.info(f"📋 Experiments to run: {experiments}")
        
        for experiment_id in experiments:
            logger.info(f"\n--- Running experiment {experiment_id} ---")
            success = self.run_experiment(experiment_id, dataset_id, phase)
            results[experiment_id] = success
        
        return results
    
    def run_experiment_on_all_datasets(self, experiment_id: str, phase: str, 
                                     checkpoint_path: str = None) -> Dict[str, bool]:
        """Run a single experiment on all datasets."""
        datasets = self.list_datasets()
        results = {}
        
        logger.info(f"🔄 Running experiment {experiment_id} on all datasets")
        logger.info(f"📊 Datasets to run: {datasets}")
        
        for dataset_id in datasets:
            logger.info(f"\n--- Running on dataset {dataset_id} ---")
            success = self.run_experiment(experiment_id, dataset_id, phase, checkpoint_path)
            results[dataset_id] = success
        
        return results
    
    def run_all_experiments_on_all_datasets(self, phase: str) -> Dict[str, Dict[str, bool]]:
        """Run all experiments on all datasets."""
        experiments = self.list_experiments()
        datasets = self.list_datasets()
        results = {}        
        logger.info(f"🌟 Running ALL experiments on ALL datasets")
        logger.info(f"📋 Experiments: {experiments}")
        logger.info(f"📊 Datasets: {datasets}")
        logger.info(f"🎯 Total combinations: {len(experiments)} × {len(datasets)} = {len(experiments) * len(datasets)}")
        
        for experiment_id in experiments:
            logger.info(f"\n=== Running experiment {experiment_id} ===")
            results[experiment_id] = {}
            
            for dataset_id in datasets:
                logger.info(f"\n--- Running {experiment_id} on {dataset_id} ---")
                success = self.run_experiment(experiment_id, dataset_id, phase)
                results[experiment_id][dataset_id] = success
        
        return results


def main():
    """Main function to parse arguments and execute experiments."""
    parser = argparse.ArgumentParser(
        description="Outil en ligne de commande pour exécuter les expériences EasyTPP",
        formatter_class=argparse.RawDescriptionHelpFormatter,        
        epilog="""
Exemples d'utilisation:
  # Lister les expériences et datasets disponibles
  python run_model.py --list-experiments
  python run_model.py --list-datasets
  
  # Exécuter une expérience spécifique
  python run_model.py --experiment THP --dataset H2expc --phase train
  python run_model.py --experiment THP --dataset H2expc --phase test --checkpoint checkpoints/model.ckpt
  
  # Exécuter toutes les phases d'une expérience
  python run_model.py --experiment THP --dataset H2expc --phase all
  
  # Exécuter toutes les expériences sur un dataset
  python run_model.py --all-experiments --dataset H2expc --phase test
  
  # Exécuter une expérience sur tous les datasets
  python run_model.py --experiment THP --all-datasets --phase test
  
  # Exécuter toutes les expériences sur tous les datasets
  python run_model.py --all-experiments --all-datasets --phase test
        """
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default='./runner_config.yaml',
                        help='Chemin vers le fichier de configuration YAML (défaut: ./runner_config.yaml)')
    
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                        help='Répertoire de sortie pour les résultats (défaut: ./experiment_results)')
    
    # Experiment selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--experiment', type=str,
                       help='ID de l\'expérience à exécuter')
    group.add_argument('--all-experiments', action='store_true',
                       help='Exécuter toutes les expériences disponibles')
      # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('--dataset', type=str,
                               help='ID du dataset à utiliser')
    dataset_group.add_argument('--all-datasets', action='store_true',
                               help='Exécuter sur tous les datasets disponibles')
    
    # Execution parameters
    parser.add_argument('--phase', type=str, default='test',
                        choices=['train', 'test', 'predict', 'all'],
                        help='Phase à exécuter (défaut: test)')
    
    parser.add_argument('--checkpoint', type=str,
                        help='Chemin vers le fichier checkpoint (optionnel)')
    
    # Listing options
    parser.add_argument('--list-experiments', action='store_true',
                        help='Lister toutes les expériences disponibles')
    
    parser.add_argument('--list-datasets', action='store_true',
                        help='Lister tous les datasets disponibles')
    
    # Debug options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Activer le mode verbeux')
    
    parser.add_argument('--debug', action='store_true',
                        help='Activer le mode debug avec traceback détaillé')

    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = ModelRunner(
            config_path=args.config,
            output_dir=args.output_dir,
            debug=args.debug or args.verbose
        )
        
        # Handle listing options
        if args.list_experiments:
            experiments = runner.list_experiments()
            print("📋 Expériences disponibles:")
            for exp in experiments:
                print(f"  - {exp}")
            return 0
        
        if args.list_datasets:
            datasets = runner.list_datasets()
            print("📊 Datasets disponibles:")
            for dataset in datasets:
                print(f"  - {dataset}")
            return 0
          # Validate required arguments
        if not (args.dataset or args.all_datasets):
            print("❌ Erreur: Vous devez spécifier un dataset avec --dataset ou --all-datasets")
            print("Utilisez --list-datasets pour voir les options disponibles")
            return 1
        
        if not (args.experiment or args.all_experiments):
            print("❌ Erreur: Vous devez spécifier --experiment ou --all-experiments")
            print("Utilisez --list-experiments pour voir les options disponibles")
            return 1
        
        # Execute experiments
        start_time = time.time()
        
        if args.all_experiments and args.all_datasets:
            # All experiments on all datasets
            logger.info(f"🌟 Executing ALL experiments on ALL datasets")
            results = runner.run_all_experiments_on_all_datasets(args.phase)
            
            # Summary
            total_combinations = 0
            successful_combinations = 0
            
            for exp, datasets_results in results.items():
                for dataset, success in datasets_results.items():
                    total_combinations += 1
                    if success:
                        successful_combinations += 1
            
            logger.info(f"� Summary: {successful_combinations}/{total_combinations} combinations successful")
            
            if successful_combinations < total_combinations:
                failed_combinations = []
                for exp, datasets_results in results.items():
                    for dataset, success in datasets_results.items():
                        if not success:
                            failed_combinations.append(f"{exp}+{dataset}")
                logger.error(f"❌ Failed combinations: {failed_combinations}")
        
        elif args.all_experiments and args.dataset:
            # All experiments on single dataset
            logger.info(f"�🚀 Executing all experiments on dataset: {args.dataset}")
            results = runner.run_all_experiments_on_dataset(args.dataset, args.phase)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"📊 Summary: {successful}/{total} experiments successful")
            
            if successful < total:
                failed_experiments = [exp for exp, success in results.items() if not success]
                logger.error(f"❌ Failed experiments: {failed_experiments}")
        
        elif args.experiment and args.all_datasets:
            # Single experiment on all datasets
            logger.info(f"🚀 Executing experiment {args.experiment} on all datasets")
            results = runner.run_experiment_on_all_datasets(args.experiment, args.phase, args.checkpoint)
            
            # Summary
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            logger.info(f"📊 Summary: {successful}/{total} datasets successful")
            
            if successful < total:
                failed_datasets = [dataset for dataset, success in results.items() if not success]
                logger.error(f"❌ Failed datasets: {failed_datasets}")
        
        else:
            # Single experiment on single dataset
            logger.info(f"🚀 Executing experiment: {args.experiment}")
            success = runner.run_experiment(
                args.experiment, 
                args.dataset, 
                args.phase, 
                args.checkpoint
            )
            
            if not success:
                return 1
        
        total_time = time.time() - start_time
        
        logger.info(f"🎉 Execution completed!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {runner.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏸️  Execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {str(e)}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
