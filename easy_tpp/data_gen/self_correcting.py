import numpy as np
from typing import Optional, List, Tuple

from easy_tpp.data_gen.base_simulator import BaseSimulator

class SelfCorrecting(BaseSimulator):
    """
    Simulateur de processus ponctuels temporels avec correction automatique.
    """
    
    def __init__(self, 
                 dim_process: int,
                 mu: float = 1.0,
                 alpha: float = 1.0,
                 start_time: float = 100,
                 end_time: float = 200,
                 seed: Optional[int] = None):
        """
        Initialise un simulateur de processus ponctuel temporel avec correction automatique.
        
        Args:
            dim_process (int): Dimension du processus (nombre de types d'événements)
            mu (float): Paramètre de taux de base
            alpha (float): Paramètre de réduction après un événement
            start_time (float): Temps de début de la simulation
            end_time (float): Temps de fin de la simulation
            seed (int, optional): Graine pour la reproductibilité
        """
        super().__init__(dim_process, start_time, end_time, seed)
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        
    def simulate(self, mu: float, alpha: float, t_end: float) -> np.ndarray:
        """
        Génère un processus auto-correctif.
        
        Args:
            mu (float): Paramètre de taux de base
            alpha (float): Paramètre de réduction après un événement
            t_end (float): Temps de fin de simulation
            
        Returns:
            np.ndarray: Array des temps d'événements
        """
        t = 0
        x = 0
        T = []
        
        while t < t_end:
            e = np.random.exponential()
            tau = np.log(e*mu/np.exp(x) + 1)/mu  # e = (np.exp(mu*tau) - 1)*np.exp(x)/mu
            t = t + tau
            
            if t >= t_end:
                break
                
            T.append(t)
            x = x + mu*tau
            x = x - alpha
        
        return np.array(T)