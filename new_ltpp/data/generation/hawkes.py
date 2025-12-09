from typing import Dict, List, Optional, Tuple

import numpy as np

from new_ltpp.data.generation.base_simulator import Simulator


class HawkesSimulator(Simulator):
    """
    Classe pour simuler des processus de Hawkes multidimensionnels.
    Un processus de Hawkes est un processus de points auto-excitant où l'occurrence
    d'événements augmente la probabilité d'événements futurs.
    """

    def __init__(
        self,
        mu: np.ndarray | List[float],
        alpha: np.ndarray | List[float],
        beta: np.ndarray | List[float],
        **kwargs,
    ):
        """
        Initialise un simulateur de processus de Hawkes.

        Args:
            mu (np.ndarray): Intensités de base pour chaque dimension [dim]
            alpha (np.ndarray): Matrice d'excitation des intensités [dim, dim]
            beta (np.ndarray): Matrice des taux de décroissance exponentielle [dim, dim]
            dim_process (int): Dimension du processus (nombre de types d'événements)
            start_time (float): Temps de début de la simulation
            end_time (float): Temps de fin de la simulation
            seed (int, optional): Graine pour la reproductibilité
        """
        # Initialisation de la classe parente
        super().__init__(**kwargs)

        # Vérification des dimensions
        if len(mu) != self.dim_process:
            raise ValueError(
                f"mu doit être de dimension {self.dim_process}, mais a {len(mu)}"
            )
        self.mu = np.array(mu).reshape(self.dim_process)
        self.alpha = np.array(alpha).reshape(self.dim_process, self.dim_process)
        self.beta = np.array(beta).reshape(self.dim_process, self.dim_process)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simule un processus de Hawkes multivarié jusqu'au temps end_time.

        Returns:
            tuple: (times, marks) où:
                - times: np.ndarray de tous les temps d'événements
                - marks: np.ndarray de tous les types/dimensions d'événements
        """
        dim = self.dim_process
        times = []
        marks = []

        # Temps actuel
        t = self.start_time
        event_count = 0

        # Matrice de contribution d'intensité initiale [to_process][from_process]
        lambda_trg = np.ones((dim, dim))

        while t < self.end_time:
            # Intensité totale pour chaque dimension
            lambda_total = np.array(
                [self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)]
            )
            lambda_sum = np.sum(lambda_total)

            # Échantillonnage du temps d'attente jusqu'au prochain événement
            dt = (
                np.random.exponential(scale=1 / lambda_sum)
                if lambda_sum > 0
                else float("inf")
            )
            t = t + dt

            if t >= self.end_time:
                break

            # Mise à jour des contributions d'intensité basée sur la décroissance exponentielle
            lambda_trg *= np.exp(
                -self.beta * dt
            )  # Décroissance exponentielle des contributions

            # Intensité totale après la décroissance
            lambda_next = np.array(
                [self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)]
            )
            lambda_next_sum = np.sum(lambda_next)

            # Test d'acceptation/rejet pour l'occurrence d'un événement
            if (
                np.random.rand() < lambda_next_sum / lambda_sum
            ):  # Acceptation de l'événement
                # Sélection aléatoire de la dimension à laquelle appartient l'événement
                event_dim = np.random.choice(dim, p=lambda_total / lambda_sum)

                # Ajout de l'événement
                times.append(t)
                marks.append(event_dim)

                # Mise à jour des contributions d'intensité
                lambda_trg[:, event_dim] += self.alpha[:, event_dim]

                event_count += 1

        # Conversion en tableaux numpy
        return np.array(times), np.array(marks)

    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur de Hawkes.

        Returns:
            Dict: Métadonnées spécifiques au simulateur de Hawkes
        """
        return {
            "hawkes_parameters": {
                "mu": self.mu.tolist(),
                "alpha": self.alpha.tolist(),
                "beta": self.beta.tolist(),
            }
        }

    def compute_theoretical_intensities(
        self, time_points: np.ndarray, event_times: np.ndarray, event_marks: np.ndarray
    ) -> np.ndarray:
        """
        Calcule les intensités théoriques du processus de Hawkes aux points temporels donnés.

        Pour un processus de Hawkes : λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_{ij} * exp(-β_{ij} * (t - t_k^j))

        Args:
            time_points (np.ndarray): Points temporels où calculer les intensités
            event_times (np.ndarray): Tous les temps d'événements
            event_marks (np.ndarray): Tous les types/dimensions d'événements

        Returns:
            np.ndarray: Matrice des intensités [len(time_points), dim_process]
        """
        intensities = np.zeros((len(time_points), self.dim_process))

        for t_idx, t in enumerate(time_points):
            for i in range(self.dim_process):
                # Intensité de base
                intensity = self.mu[i]

                # Contribution des événements passés
                for j in range(self.dim_process):
                    # Filtrer les événements passés de type j
                    mask = (event_times < t) & (event_marks == j)
                    past_events = event_times[mask]
                    if len(past_events) > 0:
                        # Somme des contributions exponentielles décroissantes
                        contributions = self.alpha[i, j] * np.exp(
                            -self.beta[i, j] * (t - past_events)
                        )
                        intensity += np.sum(contributions)

                intensities[t_idx, i] = max(intensity, 0)  # Intensité positive

        return intensities
