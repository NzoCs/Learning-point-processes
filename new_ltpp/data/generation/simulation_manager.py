from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


class SimulationManager:
    """
    Gestionnaire de simulations qui orchestre la génération et le formatage des données.
    Prend une fonction de simulation et gère le bulk processing et le formatage.
    """

    def __init__(
        self,
        simulation_func: Callable[[], Tuple[np.ndarray, np.ndarray]],
        dim_process: int,
        start_time: float,
        end_time: float,
    ):
        """
        Initialise le gestionnaire de simulations.

        Args:
            simulation_func: Fonction qui simule un processus et retourne (times, marks)
            dim_process: Dimension du processus
            start_time: Temps de début de la simulation
            end_time: Temps de fin de la simulation
        """
        self.simulation_func = simulation_func
        self.dim_process = dim_process
        self.start_time = start_time
        self.end_time = end_time

    def bulk_simulate(self, num_simulations: int) -> List[Dict]:
        """
        Génère plusieurs simulations et les formate.

        Args:
            num_simulations: Nombre de simulations à générer

        Returns:
            Liste des simulations formatées
        """
        simulations = []

        for _ in tqdm(
            range(num_simulations), desc=f"Simulation de {num_simulations} processus"
        ):
            times, marks = self.simulation_func()
            simulations.append((times, marks))

        # Format simulations for dataset
        formatted_data = self.format_simulations(simulations)

        return formatted_data

    def format_simulations(
        self, simulations: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """
        Formate les simulations au format dataset Hugging Face.

        Args:
            simulations: Liste de tuples (times, marks) issus de simulate()

        Returns:
            Liste de dictionnaires, chacun représentant une séquence
        """
        formatted_data = []

        for seq_idx, (times, marks) in enumerate(simulations):
            # Filtrer les timestamps supérieurs à start_time
            mask = times > self.start_time
            valid_times = times[mask]
            valid_marks = marks[mask]

            if len(valid_times) == 0:
                continue

            # Trier par temps (devrait déjà être trié, mais par sécurité)
            sort_idx = np.argsort(valid_times)
            sorted_times = valid_times[sort_idx]
            sorted_marks = valid_marks[sort_idx]

            # Calculer time since start et time differences
            time_since_start = sorted_times - sorted_times[0]
            time_since_last_event = np.diff(sorted_times, prepend=sorted_times[0])

            temp_dict = {
                "dim_process": self.dim_process,
                "seq_len": len(sorted_times),
                "seq_idx": seq_idx,
                "time_since_start": time_since_start.tolist(),
                "time_since_last_event": time_since_last_event.tolist(),
                "type_event": sorted_marks.tolist(),
            }
            formatted_data.append(temp_dict)

        return formatted_data
