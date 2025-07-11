import numpy as np
from typing import List, Tuple, Dict, Optional

from easy_tpp.data.generation.base_simulator import BaseSimulator


class HawkesSimulator(BaseSimulator):
    """
    Classe pour simuler des processus de Hawkes multidimensionnels.
    Un processus de Hawkes est un processus de points auto-excitant où l'occurrence
    d'événements augmente la probabilité d'événements futurs.
    """
    
    def __init__(self, 
                 mu: List[float], #dim_process
                 alpha: List[List[float]], #dim_process x dim_process
                 beta: List[List[float]], #dim_process x dim_process
                 dim_process: int,
                 start_time: float = 100,
                 end_time: float = 200,
                 nb_events: int = float('inf'),
                 seed: Optional[int] = None):
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
        super().__init__(dim_process, start_time, end_time, nb_events, seed)

        # Vérification des dimensions
        if len(mu) != dim_process:
            raise ValueError(f"mu doit être de dimension {dim_process}, mais a {len(mu)}")
        self.mu = np.array(mu).reshape(dim_process)
        self.alpha = np.array(alpha).reshape(dim_process, dim_process)
        self.beta = np.array(beta).reshape(dim_process, dim_process)


    def simulate(self) -> Tuple[List[np.ndarray]]:
        """
        Simule un processus de Hawkes multivarié jusqu'au temps end_time.
        
        Returns:
            tuple: Liste d'arrays de temps d'événements pour chaque dimension
        """
        dim = self.dim_process
        events = [[] for _ in range(dim)]
        
        # Temps actuel
        t = self.start_time
        event_count = 0

        # Matrice de contribution d'intensité initiale [to_process][from_process]
        lambda_trg = np.zeros((dim, dim))

        while t < self.end_time:
            # Intensité totale pour chaque dimension
            lambda_total = np.array([self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
            lambda_sum = np.sum(lambda_total)
            
            # Échantillonnage du temps d'attente jusqu'au prochain événement
            dt = np.random.exponential(scale=1/lambda_sum) if lambda_sum > 0 else float('inf')
            t = t + dt
            
            if t >= self.end_time:
                break
                
            # Mise à jour des contributions d'intensité basée sur la décroissance exponentielle
            lambda_trg *= np.exp(-self.beta * dt.unsqueeze(0).unsqueeze(0))  # Décroissance exponentielle des contributions

            # Intensité totale après la décroissance
            lambda_next = np.array([self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
            lambda_next_sum = np.sum(lambda_next)
            
            # Test d'acceptation/rejet pour l'occurrence d'un événement
            if np.random.rand() < lambda_next_sum / lambda_sum:  # Acceptation de l'événement
                # Sélection aléatoire de la dimension à laquelle appartient l'événement
                event_dim = np.random.choice(dim, p=lambda_total/lambda_sum)
                
                # Ajout de l'événement à la dimension correspondante
                events[event_dim].append(t)
                
                # Mise à jour des contributions d'intensité
                lambda_trg[:, event_dim] += self.alpha[:, event_dim]

                event_count += 1
                if event_count >= self.nb_events:
                    return tuple(np.array(events_dim) for events_dim in events)
        
        # Conversion en tableaux numpy
        return tuple(np.array(events_dim) for events_dim in events)
    
    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur de Hawkes.
        
        Returns:
            Dict: Métadonnées spécifiques au simulateur de Hawkes
        """
        return {
            'hawkes_parameters': {
                'mu': self.mu.tolist(),
                'alpha': self.alpha.tolist(),
                'beta': self.beta.tolist()
            }
        }
    
    def compute_theoretical_intensities(self, 
                                      time_points: np.ndarray, 
                                      events_by_dim: Tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Calcule les intensités théoriques du processus de Hawkes aux points temporels donnés.
        
        Pour un processus de Hawkes : λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_{ij} * exp(-β_{ij} * (t - t_k^j))
        
        Args:
            time_points (np.ndarray): Points temporels où calculer les intensités
            events_by_dim (Tuple[np.ndarray, ...]): Événements générés par dimension
            
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
                    past_events = events_by_dim[j][events_by_dim[j] < t]
                    if len(past_events) > 0:
                        # Somme des contributions exponentielles décroissantes
                        contributions = self.alpha[i, j] * np.exp(-self.beta[i, j] * (t - past_events))
                        intensity += np.sum(contributions)
                
                intensities[t_idx, i] = max(intensity, 0)  # Intensité positive
        
        return intensities