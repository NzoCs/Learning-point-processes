from typing import Dict, List, Optional, Tuple

import numpy as np

from new_ltpp.data.generation.base_simulator import Simulator


class SelfCorrecting(Simulator):
    """
    Simulateur de processus ponctuels temporels avec correction automatique multivariable.
    Cette version permet aux événements de différentes dimensions d'avoir une influence les uns sur les autres.
    """

    def __init__(
        self,
        mu: np.ndarray | List[float],
        alpha: np.ndarray | List[List[float]],
        **kwargs,
    ):
        """
        Initialise un simulateur de processus ponctuel temporel avec correction automatique multivariable.

        Args:
            mu (np.ndarray | List[float]): Paramètre de taux de base (scalaire ou array)
            alpha (np.ndarray | List[List[float]]): Matrice d'influence entre dimensions.
                                               alpha_matrix[i,j] indique l'influence d'un événement de type j sur le taux de la dimension i.
                                               Si None, une matrice identité est utilisée (pas d'influence croisée).
        """
        super().__init__(**kwargs)

        # Support both scalar and array inputs for mu
        if isinstance(mu, list):
            self.mu = np.array(mu).reshape(self.dim_process)
        else:
            self.mu = np.full(self.dim_process, mu)

        if isinstance(alpha, list):
            self.alpha_matrix = np.array(alpha).reshape(
                self.dim_process, self.dim_process
            )
        else:
            self.alpha_matrix = np.full((self.dim_process, self.dim_process), alpha)

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

    def batch_simulate(
        self, num_simulations: int, batch_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate multiple independent Self-Correcting processes in parallel.

        Unlike Hawkes, this simulator uses exact inversion (no thinning/rejection):
        for each dimension, the next event time is computed analytically, then the
        minimum over dimensions is selected. The batch version vectorizes this over
        B independent paths simultaneously.

        Args:
            num_simulations: Total number of paths to generate.
            batch_size: Number of paths to process in parallel (defaults to all at once).

        Returns:
            List of (times, marks) tuples, one per path.
        """
        if batch_size is None:
            batch_size = num_simulations

        results: List[Tuple[np.ndarray, np.ndarray]] = []
        remaining = num_simulations

        while remaining > 0:
            B = min(batch_size, remaining)
            results.extend(self._simulate_batch(B))
            remaining -= B

        return results

    def _simulate_batch(
        self, B: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Core vectorized exact-sampling loop for a batch of B independent paths.

        State tensors:
            x                (B, dim)  — compensator state per path and dimension
            t                (B,)      — current time per path
            next_event_times (B, dim)  — next scheduled event per path and dimension
            active           (B,)      — True while path has not yet reached end_time

        At each step:
          1. Find the next event (argmin over dim axis) — vectorized over (B, dim)
          2. Advance x for ALL dimensions — vectorized
          3. Record the winning (path, dim) pairs
          4. Apply alpha correction — vectorized scatter via np.add.at
          5. Recompute tau for the winning dimension only — vectorized

        Args:
            B: Number of paths in this batch.

        Returns:
            List of (times, marks) tuples of length B.
        """
        dim = self.dim_process

        # State variables
        x = np.zeros((B, dim), dtype=np.float64)                  # (B, dim)
        t = np.full(B, self.start_time, dtype=np.float64)         # (B,)
        next_event_times = np.full((B, dim), np.inf, dtype=np.float64)  # (B, dim)
        active = np.ones(B, dtype=bool)                            # (B,)

        # Event storage
        times_list: List[List[float]] = [[] for _ in range(B)]
        marks_list: List[List[int]] = [[] for _ in range(B)]

        # ── Initialise first next-event times for each (path, dim) ────────────
        # tau = log(E * mu[d] / exp(x[:, d]) + 1) / mu[d]   with E ~ Exp(1)
        E = np.random.exponential(size=(B, dim))  # (B, dim)
        tau_init = np.log(
            E * self.mu[np.newaxis, :] / np.exp(x) + 1
        ) / self.mu[np.newaxis, :]                # (B, dim)
        next_event_times = t[:, np.newaxis] + tau_init  # (B, dim)

        while np.any(active):
            # ── Find next event per path ──────────────────────────────────────
            next_dim = np.argmin(next_event_times, axis=1)   # (B,) — winning dimension
            next_time = next_event_times[np.arange(B), next_dim]  # (B,)

            # Paths whose next event overshoots end_time become inactive
            will_end = next_time >= self.end_time
            active &= ~will_end

            if not np.any(active):
                break

            # ── Advance compensator state x for active paths ──────────────────
            # x[b, :] += mu * (next_time[b] - t[b])
            delta_t = np.where(active, next_time - t, 0.0)   # (B,)
            x += self.mu[np.newaxis, :] * delta_t[:, np.newaxis]  # (B, dim)

            # ── Record events for active paths ────────────────────────────────
            active_idx = np.where(active)[0]
            for b in active_idx:
                times_list[b].append(float(next_time[b]))
                marks_list[b].append(int(next_dim[b]))

            # ── Apply alpha correction: x[b, :] -= alpha_matrix[:, next_dim[b]] ──
            # Flatten: for each active path b, subtract alpha_matrix[:, d]
            # alpha_matrix[:, next_dim[active_idx]] has shape (dim, n_active)
            # We want x[active_idx, :] -= alpha_matrix[:, next_dim[active_idx]].T
            winning_dims = next_dim[active_idx]              # (n_active,)
            x[active_idx] -= self.alpha_matrix[:, winning_dims].T  # (n_active, dim)

            # ── Update current time ───────────────────────────────────────────
            t = np.where(active, next_time, t)

            # ── Recompute next-event time ONLY for the winning dimension ──────
            # Only active paths need a new tau for their winning dimension
            E_new = np.random.exponential(size=len(active_idx))  # (n_active,)
            x_d = x[active_idx, winning_dims]                    # (n_active,)
            mu_d = self.mu[winning_dims]                          # (n_active,)
            tau_new = np.log(E_new * mu_d / np.exp(x_d) + 1) / mu_d  # (n_active,)
            next_event_times[active_idx, winning_dims] = t[active_idx] + tau_new

            # Inactive paths: push their next_event_times to inf so they're never selected
            if not np.all(active):
                next_event_times[~active] = np.inf

        return [
            (np.array(times_list[b]), np.array(marks_list[b]))
            for b in range(B)
        ]

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
