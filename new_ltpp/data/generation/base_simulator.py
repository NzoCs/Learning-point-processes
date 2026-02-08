from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Protocol, runtime_checkable

import numpy as np


class Simulator(ABC):
    """
    Classe de base pour tous les simulateurs de processus ponctuels temporels.
    Responsabilité unique : simuler un processus et retourner (times, marks).
    """

    def __init__(
        self,
        dim_process: int,
        start_time: float = 100,
        end_time: float = 200,
        seed: Optional[int] = None,
    ):
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

        if seed is not None:
            import random

            np.random.seed(seed)
            random.seed(seed)

    @abstractmethod
    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule un processus ponctuel temporel.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, marks) où:
                - times: np.ndarray de tous les temps d'événements
                - marks: np.ndarray de tous les types/dimensions d'événements
        """
        pass

    def get_metadata(self, num_simulations: int) -> Dict:
        """
        Renvoie les métadonnées du simulateur.

        Args:
            num_simulations (int): Nombre de simulations générées

        Returns:
            Dict: Métadonnées complètes incluant info de base et spécifiques au simulateur
        """
        metadata = {
            "simulation_info": {
                "num_simulations": num_simulations,
                "dimension": self.dim_process,
                "time_interval": [self.start_time, self.end_time],
                "simulator_type": self.__class__.__name__,
            }
        }

        # Ajout des métadonnées spécifiques au simulateur
        simulator_metadata = self.get_simulator_metadata()
        if simulator_metadata:
            metadata.update(simulator_metadata)

        return metadata

    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.
        À surcharger par les sous-classes pour ajouter des métadonnées spécifiques.

        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {}  # Par défaut, pas de métadonnées spécifiques

    def generate_and_save(
        self,
        output_dir: str,
        num_simulations: int,
        splits: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Génère plusieurs simulations et les sauvegarde dans des fichiers.

        Args:
            output_dir: Répertoire de sortie
            num_simulations: Nombre de simulations à générer
            splits: Dictionnaire des splits (train/test/dev), par défaut {"train": 0.6, "test": 0.2, "dev": 0.2}
        """
        import json
        from pathlib import Path

        if splits is None:
            splits = {"train": 0.6, "test": 0.2, "dev": 0.2}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Générer toutes les simulations
        all_sequences = []
        for _ in range(num_simulations):
            times, marks = self.simulate()
            all_sequences.append(
                {
                    "time_seqs": times.tolist(),
                    "type_seqs": marks.tolist(),
                }
            )

        # Diviser selon les splits
        n_total = len(all_sequences)
        indices = np.random.permutation(n_total)

        current_idx = 0
        for split_name, ratio in splits.items():
            n_split = int(n_total * ratio)
            split_indices = indices[current_idx : current_idx + n_split]
            current_idx += n_split

            split_data = [all_sequences[i] for i in split_indices]

            # Sauvegarder le split
            split_file = output_path / f"{split_name}.json"
            with open(split_file, "w") as f:
                json.dump(split_data, f, indent=2)

        # Sauvegarder les métadonnées
        metadata = self.get_metadata(num_simulations)
        metadata["splits"] = splits
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    @abstractmethod
    def compute_theoretical_intensities(
        self, time_points: np.ndarray, event_times: np.ndarray, event_marks: np.ndarray
    ) -> np.ndarray:
        """
        Calcule les intensités théoriques aux points temporels donnés.

        Args:
            time_points (np.ndarray): Points temporels où calculer les intensités
            event_times (np.ndarray): Tous les temps d'événements
            event_marks (np.ndarray): Tous les types/dimensions d'événements

        Returns:
            np.ndarray: Matrice des intensités [len(time_points), dim_process]
        """
        pass


@runtime_checkable
class ISimulator(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    dim_process: int
    start_time: float
    end_time: float

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a temporal point process.

        Returns:
            Tuple of (times, marks)
        """
        ...

    def compute_theoretical_intensities(
        self,
        time_points: np.ndarray,
        event_times: np.ndarray,
        event_marks: np.ndarray,
    ) -> np.ndarray:
        """Compute theoretical intensities at given time points."""
        ...

    def get_metadata(self, num_simulations: int) -> Dict:
        """Return metadata of the simulator."""
        ...
