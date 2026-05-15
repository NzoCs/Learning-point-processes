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
        alpha: np.ndarray | List[List[float]],
        beta: np.ndarray | List[List[float]],
        **kwargs,
    ):
        """
        Initialise un simulateur de processus de Hawkes.

        Args:
            mu (np.ndarray | List[float]): Intensités de base pour chaque dimension [dim]
            alpha (np.ndarray | List[List[float]]): Matrice d'excitation des intensités [dim, dim]
            beta (np.ndarray | List[List[float]]): Matrice des taux de décroissance exponentielle [dim, dim]
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

    def batch_simulate(
        self, num_simulations: int, batch_size: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate multiple independent Hawkes processes in parallel using vectorized
        numpy operations over a batch dimension.

        At each iteration of the thinning algorithm, ALL paths are advanced
        simultaneously:
          - dt sampling      : vectorized over (B,)
          - exponential decay: vectorized over (B, dim, dim)
          - acceptance test  : vectorized over (B,)
          - dim sampling     : vectorized multinomial via cumsum on (n_acc, dim)
          - lambda_trg update: np.add.at over (n_acc * dim,) flat indices

        Event recording uses Python lists (variable-length sequences per path).

        Args:
            num_simulations: Total number of independent paths to generate.
            batch_size: Number of paths to process in parallel. Defaults to
                        num_simulations (one single vectorized batch).

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
        Core vectorized thinning loop for a batch of B independent paths.

        Args:
            B: Number of paths in this batch.

        Returns:
            List of (times, marks) tuples of length B.
        """
        dim = self.dim_process

        # Current time for each path  (B,)
        t = np.full(B, self.start_time, dtype=np.float64)

        # active[b] = True while path b has not yet reached end_time
        active = np.ones(B, dtype=bool)

        # lambda_trg[b, i, j] = excitation contribution to dim i from dim j in path b
        # Initialised to 1 (same as scalar simulate)
        lambda_trg = np.ones((B, dim, dim), dtype=np.float64)

        # Event storage — variable-length lists per path
        times_list: List[List[float]] = [[] for _ in range(B)]
        marks_list: List[List[int]] = [[] for _ in range(B)]

        while np.any(active):
            # ── Upper-bound intensity ─────────────────────────────────────────
            # lambda_total[b, i] = mu[i] + sum_j lambda_trg[b, i, j]
            lambda_total = self.mu[np.newaxis, :] + lambda_trg.sum(axis=2)  # (B, dim)
            lambda_sum = lambda_total.sum(axis=1)  # (B,)

            # Guard: paths with zero total rate cannot generate events
            safe_rate = np.where(lambda_sum > 0, lambda_sum, 1.0)

            # ── Sample waiting times ──────────────────────────────────────────
            dt = np.random.exponential(scale=1.0 / safe_rate)  # (B,)
            dt = np.where(lambda_sum > 0, dt, np.inf)

            t_new = t + dt

            # Paths that overshoot end_time become inactive
            will_end = t_new >= self.end_time
            active &= ~will_end

            if not np.any(active):
                break

            t = np.where(active, t_new, t)

            # ── Exponential decay of contributions ────────────────────────────
            # decay[b, i, j] = exp(-beta[i, j] * dt[b])
            decay = np.exp(
                -self.beta[np.newaxis, :, :] * dt[:, np.newaxis, np.newaxis]
            )  # (B, dim, dim)
            lambda_trg *= decay

            # ── Post-decay intensity ──────────────────────────────────────────
            lambda_next = self.mu[np.newaxis, :] + lambda_trg.sum(axis=2)  # (B, dim)
            lambda_next_sum = lambda_next.sum(axis=1)  # (B,)

            # ── Acceptance test ───────────────────────────────────────────────
            u_accept = np.random.rand(B)
            accepted = active & (u_accept < lambda_next_sum / safe_rate)  # (B,)

            n_acc = int(accepted.sum())
            if n_acc == 0:
                continue

            accepted_idx = np.where(accepted)[0]  # (n_acc,)

            # ── Vectorized multinomial: sample event dimension ────────────────
            # p[k, i] = prob that path accepted_idx[k] fires dimension i
            p = lambda_total[accepted_idx] / lambda_sum[accepted_idx, np.newaxis]  # (n_acc, dim)
            cumsum = np.cumsum(p, axis=1)  # (n_acc, dim)
            u_dim = np.random.rand(n_acc, 1)
            event_dims = (u_dim > cumsum).sum(axis=1)              # (n_acc,)
            event_dims = np.minimum(event_dims, dim - 1).astype(int)  # numerical guard

            # ── Record events ─────────────────────────────────────────────────
            for k, b in enumerate(accepted_idx):
                times_list[b].append(float(t[b]))
                marks_list[b].append(int(event_dims[k]))

            # ── Update lambda_trg via np.add.at ───────────────────────────────
            # For each accepted event (b, d): lambda_trg[b, :, d] += alpha[:, d]
            # Expand to flat index triplets: (b_rep, i_rep, d_rep)
            b_rep = np.repeat(accepted_idx, dim)        # (n_acc * dim,)
            i_rep = np.tile(np.arange(dim), n_acc)     # (n_acc * dim,)
            d_rep = np.repeat(event_dims, dim)          # (n_acc * dim,)
            np.add.at(lambda_trg, (b_rep, i_rep, d_rep), self.alpha[i_rep, d_rep])

        return [
            (np.array(times_list[b]), np.array(marks_list[b]))
            for b in range(B)
        ]

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
