from typing import Dict, List, Optional, Tuple

import numpy as np

from new_ltpp.data.generation.base_simulator import Simulator


class SelfCorrecting(Simulator):
    """
    Simulateur de processus ponctuels temporels avec correction automatique.
    """

    def __init__(self, mu: float, alpha: float, **kwargs):
        """
        Initialise un simulateur de processus ponctuel temporel avec correction automatique.

        Args:
            mu (float): Paramètre de taux de base
            alpha (float): Paramètre de réduction après un événement
        """
        super().__init__(**kwargs)

        # Support both scalar and array inputs for mu and alpha
        if isinstance(mu, (int, float)):
            self.mu = np.array([mu] * self.dim_process)
        else:
            self.mu = np.array(mu)

        if isinstance(alpha, (int, float)):
            self.alpha = np.array([alpha] * self.dim_process)
        else:
            self.alpha = np.array(alpha)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un processus auto-correctif pour chaque dimension.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, marks) où:
                - times: np.ndarray de tous les temps d'événements
                - marks: np.ndarray de tous les types/dimensions d'événements
        """
        all_times = []
        all_marks = []

        for dim in range(self.dim_process):
            # Simulate each dimension independently
            t = self.start_time
            x = 0

            while t < self.end_time:
                e = np.random.exponential()
                tau = np.log(e * self.mu[dim] / np.exp(x) + 1) / self.mu[dim]
                t = t + tau

                if t >= self.end_time:
                    break

                all_times.append(t)
                all_marks.append(dim)
                x = x + self.mu[dim] * tau
                x = x - self.alpha[dim]

        # Sort by time
        times = np.array(all_times)
        marks = np.array(all_marks)
        sort_idx = np.argsort(times)

        return times[sort_idx], marks[sort_idx]

    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.

        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {
            "self_correcting_parameters": {
                "mu": self.mu.tolist(),
                "alpha": self.alpha.tolist(),
            }
        }

    def compute_theoretical_intensities(
        self, time_points: np.ndarray, event_times: np.ndarray, event_marks: np.ndarray
    ) -> np.ndarray:
        """
        Calcule les intensités théoriques du processus auto-correctif aux points temporels donnés.

        Pour un processus auto-correctif : λ_i(t) = μ_i * exp(x_i(t))
        où x_i(t) = μ_i * (t - t_last) - α_i * N_i(t)

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
                # Événements passés pour cette dimension
                mask = (event_times < t) & (event_marks == i)
                past_events = event_times[mask]

                # Temps du dernier événement (ou start_time si aucun événement)
                if len(past_events) > 0:
                    t_last = past_events[-1]
                    num_events = len(past_events)
                else:
                    t_last = self.start_time
                    num_events = 0

                # Calcul de x(t)
                x_t = self.mu[i] * (t - t_last) - self.alpha[i] * num_events

                # Intensité
                intensity = self.mu[i] * np.exp(x_t)
                intensities[t_idx, i] = max(intensity, 0)  # Intensité positive

        return intensities


class MultivariableSelfCorrecting(Simulator):
    """
    Simulateur de processus ponctuels temporels avec correction automatique multivariable.
    Cette version permet aux événements de différentes dimensions d'avoir une influence les uns sur les autres.
    """

    def __init__(
        self, mu: float = 1.0, alpha_matrix: Optional[np.ndarray] = None, **kwargs
    ):
        """
        Initialise un simulateur de processus ponctuel temporel avec correction automatique multivariable.

        Args:
            mu (float): Paramètre de taux de base (scalaire ou array)
            alpha_matrix (np.ndarray, optional): Matrice d'influence entre dimensions.
                                               alpha_matrix[i,j] indique l'influence d'un événement de type j sur le taux de la dimension i.
                                               Si None, une matrice identité est utilisée (pas d'influence croisée).
        """
        super().__init__(**kwargs)

        # Support both scalar and array inputs for mu
        if isinstance(mu, (int, float)):
            self.mu = np.array([mu] * self.dim_process)
        else:
            self.mu = np.array(mu)

        # Initialize alpha_matrix
        if alpha_matrix is None:
            # Default: identity matrix (no cross-influence)
            self.alpha_matrix = np.eye(self.dim_process)
        else:
            if alpha_matrix.shape != (self.dim_process, self.dim_process):
                raise ValueError(
                    f"alpha_matrix must be of shape ({self.dim_process}, {self.dim_process})"
                )
            self.alpha_matrix = alpha_matrix

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un processus auto-correctif multivariable.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, marks) où:
                - times: np.ndarray de tous les temps d'événements
                - marks: np.ndarray de tous les types/dimensions d'événements
        """
        # Initialize state variables for each dimension
        x = np.zeros(self.dim_process)
        t = self.start_time
        next_event_times = np.full(self.dim_process, np.inf)
        all_times = []
        all_marks = []

        # Generate initial next event times for each dimension
        for dim in range(self.dim_process):
            e = np.random.exponential()
            tau = np.log(e * self.mu[dim] / np.exp(x[dim]) + 1) / self.mu[dim]
            next_event_times[dim] = t + tau

        # Main simulation loop
        while np.min(next_event_times) < self.end_time:
            # Find the next event dimension and time
            next_dim = np.argmin(next_event_times)
            next_time = next_event_times[next_dim]

            # Update state variables for all dimensions
            delta_t = next_time - t
            x += self.mu * delta_t

            # Record the event
            all_times.append(next_time)
            all_marks.append(next_dim)

            # Apply the influence of the event on all dimensions
            x -= self.alpha_matrix[:, next_dim]

            # Generate next event time for the dimension that just had an event
            e = np.random.exponential()
            tau = (
                np.log(e * self.mu[next_dim] / np.exp(x[next_dim]) + 1)
                / self.mu[next_dim]
            )
            next_event_times[next_dim] = next_time + tau

            # Update current time
            t = next_time

        # Convert to numpy arrays (already sorted by time due to simulation logic)
        return np.array(all_times), np.array(all_marks)

    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.

        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {
            "multivariable_self_correcting_parameters": {
                "mu": self.mu.tolist(),
                "alpha_matrix": self.alpha_matrix.tolist(),
            }
        }

    def compute_theoretical_intensities(
        self, time_points: np.ndarray, event_times: np.ndarray, event_marks: np.ndarray
    ) -> np.ndarray:
        """
        Calcule les intensités théoriques du processus auto-correctif multivariable.

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
                # État x_i(t) basé sur tous les événements passés
                x_i = 0

                # Contribution de chaque dimension
                for j in range(self.dim_process):
                    mask = (event_times < t) & (event_marks == j)
                    past_events = event_times[mask]

                    if len(past_events) > 0:
                        # Temps depuis le dernier événement de la dimension j
                        t_last = past_events[-1]
                        x_i += self.mu[i] * (t - t_last)

                        # Réduction due aux événements de la dimension j
                        x_i -= self.alpha_matrix[i, j] * len(past_events)
                    else:
                        # Pas d'événements dans cette dimension
                        x_i += self.mu[i] * (t - self.start_time)

                # Intensité
                intensity = self.mu[i] * np.exp(x_i)
                intensities[t_idx, i] = max(intensity, 0)  # Intensité positive

        return intensities
