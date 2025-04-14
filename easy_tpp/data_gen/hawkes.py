import numpy as np
import json
import os
import random
from typing import List, Tuple, Dict, Optional, Union

from easy_tpp.data_gen.base_simulator import BaseSimulator


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
        super().__init__(dim_process, start_time, end_time, seed)
        
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
            for i in range(dim):
                for j in range(dim):
                    lambda_trg[i, j] *= np.exp(-self.beta[i, j] * dt)
            
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
                for i in range(dim):
                    lambda_trg[i, event_dim] += self.alpha[i, event_dim]
        
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