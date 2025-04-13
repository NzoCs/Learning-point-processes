from abc import ABC, abstractmethod
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm


class BaseSimulator(ABC):
    """
    Classe de base pour tous les simulateurs de processus ponctuels temporels.
    """
    
    def __init__(self, 
                 dim_process: int,
                 start_time: float = 0.0,
                 end_time: float = 100.0,
                 output_dir: str = 'data',
                 seed: Optional[int] = None):
        """
        Initialise un simulateur de base.
        
        Args:
            dim_process (int): Dimension du processus (nombre de types d'événements)
            start_time (float): Temps de début de la simulation
            end_time (float): Temps de fin de la simulation
            seed (int, optional): Graine pour la reproductibilité
        """
        self.dim_process = dim_process
        self.start_time = start_time
        self.end_time = end_time
        self.seed = seed
        self.output_dir = output_dir
        
        if seed is not None:
            import random
            np.random.seed(seed)
            random.seed(seed)
        
        self.simulations = None
    
    def generate_model_from_config(config: Dict) -> 'BaseSimulator':
        """
        Génère un simulateur à partir d'une configuration.
        
        Args:
            config (Dict): Configuration du simulateur
        
        Returns:
            BaseSimulator: Instance du simulateur configuré
        """
        simulator_type = config.get('simulator_type', 'Hawkes')
        
        if simulator_type == 'Hawkes':
            return HawkesSimulator(**config['simulator_params'])
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")
    
    @abstractmethod
    def simulate(self) -> None:
        """
        Simule un processus ponctuel temporel.
        
        Returns:
            Tuple[List[np.ndarray]]: Liste d'arrays de temps d'événements pour chaque dimension
        """
        pass

    def format_simulations(self):
        """
        Format multivariate Hawkes simulations to the Hugging Face dataset format.
        
        Args:
            simulations (list): List of tuples, each containing arrays of timestamps for each dimension
            dim_process (int): Number of dimensions in the process
            start_time (float): Only include timestamps greater than this value
            
        Returns:
            list: A list of dictionaries, each representing a sequence
        """
        formatted_data = []
        
        simulations = self.simulations
        dim_process = self.dim_process
        start_time = self.start_time
        
        if simulations is None:
            raise ValueError("No simulations to format. Please run the simulate method first.")
        
        for seq_idx, sim in enumerate(tqdm(simulations, desc=f"Formatting {dim_process}D simulations")):
            # Merge timestamps from all dimensions with their type
            all_timestamps = []
            all_types = []
            all_time_diff = []
            
            for dim, timestamps in enumerate(sim):
                # Filter timestamps greater than start_time
                valid_timestamps = timestamps[timestamps > start_time]
                
                if len(valid_timestamps) > 0:
                    all_timestamps.extend(valid_timestamps)
                    all_types.extend([dim] * len(valid_timestamps))
                    all_time_diff.extend(np.diff(np.array(valid_timestamps), prepend=valid_timestamps[0]))
            
            if len(all_timestamps) == 0:
                continue
            
            # Convert to numpy arrays and sort by time
            all_timestamps = np.array(all_timestamps)
            all_types = np.array(all_types)
            all_time_diff = np.array(all_time_diff)
            sort_idx = np.argsort(all_timestamps)
            sorted_timestamps = all_timestamps[sort_idx]
            sorted_types = all_types[sort_idx].tolist()
            
            # Calculate time since start and time differences
            time_since_start = sorted_timestamps - sorted_timestamps[0]
            time_since_last_event = all_time_diff[sort_idx]
                
            temp_dict = {
                'dim_process': dim_process,
                'seq_len': len(sorted_timestamps),
                'seq_idx': seq_idx,
                'time_since_start': time_since_start.tolist(),
                'time_since_last_event': time_since_last_event.tolist(),
                'type_event': sorted_types
            }
            formatted_data.append(temp_dict)
        
        return formatted_data
    
    def bulk_simulate(self, num_simulations: int) -> List[Dict]:
        """
        Simule plusieurs processus et les formate.
        
        Args:
            num_simulations (int): Nombre de simulations à générer
            
        Returns:
            list: Liste de dictionnaires, chacun représentant une séquence formatée
        """
        formatted_data = []
        
        for seq_idx in tqdm(range(num_simulations), desc=f"Génération de {num_simulations} simulations {self.dim_process}D"):
            # Simulation d'un processus
            sim = self.simulate()
            
            # Formatage de la simulation
            formatted_sim = self.format_simulations(sim)
            
            if formatted_sim:
                formatted_sim['seq_idx'] = seq_idx
                formatted_data.append(formatted_sim)
        
        return formatted_data
    
    def generate_and_save(self, 
                         num_simulations: int = 1000, 
                         output_dir: str = 'data',
                         splits: Dict[str, float] = {'train': 0.6, 'test': 0.2, 'dev': 0.2}) -> None:
        """
        Génère et sauvegarde des simulations.
        
        Args:
            num_simulations (int): Nombre de simulations à générer
            output_dir (str): Dossier où sauvegarder les fichiers de sortie
            splits (dict): Ratios pour la division train/test/dev
        """
        # Vérification que les ratios somment à 1
        assert abs(sum(splits.values()) - 1.0) < 1e-10, "Les ratios doivent sommer à 1"
        
        # Création du dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Génération des données
        print(f"Génération de {num_simulations} simulations {self.dim_process}D...")
        formatted_data = self.bulk_simulate(num_simulations)
        
        # Division des données
        print("Division des données en ensembles train/test/dev...")
        n = len(formatted_data)
        
        data_splits = {}
        start_idx = 0
        for split_name, ratio in splits.items():
            split_size = int(n * ratio)
            data_splits[split_name] = formatted_data[start_idx:start_idx + split_size]
            start_idx += split_size
        
        # Sauvegarde des données
        print("Sauvegarde des données...")
        for split_name, data in data_splits.items():
            with open(os.path.join(output_dir, f'{split_name}.json'), 'w') as f:
                json.dump(data, f, indent=2)
        
        # Sauvegarde des métadonnées génériques
        self._save_metadata(output_dir, num_simulations, data_splits, splits, formatted_data)
    
    def _save_metadata(self, 
                     output_dir: str, 
                     num_simulations: int,
                     data_splits: Dict[str, List[Dict]],
                     splits: Dict[str, float],
                     formatted_data: List[Dict]) -> None:
        """
        Sauvegarde les métadonnées des simulations.
        Cette méthode utilise get_simulator_metadata pour permettre aux sous-classes
        d'ajouter leurs propres métadonnées spécifiques.
        
        Args:
            output_dir: Répertoire de sortie
            num_simulations: Nombre de simulations
            data_splits: Données divisées par split
            splits: Ratios de division
            formatted_data: Données formatées
        """
        # Métadonnées de base communes à tous les simulateurs
        metadata = {
            'simulation_info': {
                'num_simulations': num_simulations,
                'dimension': self.dim_process,
                'time_interval': [self.start_time, self.end_time],
                'simulator_type': self.__class__.__name__
            },
            'split_info': {
                **{f'{split_name}_size': len(data) for split_name, data in data_splits.items()},
                **{f'{split_name}_ratio': ratio for split_name, ratio in splits.items()}
            },
            'total_events': sum(item['seq_len'] for item in formatted_data)
        }
        
        # Ajout des métadonnées spécifiques au simulateur
        simulator_metadata = self.get_simulator_metadata()
        if simulator_metadata:
            metadata.update(simulator_metadata)
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Toutes les données ont été sauvegardées dans {output_dir}")
    
    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.
        À surcharger par les sous-classes pour ajouter des métadonnées spécifiques.
        
        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {}  # Par défaut, pas de métadonnées spécifiques
