import numpy as np
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

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
        
        # Support both scalar and array inputs for mu and alpha
        if isinstance(mu, (int, float)):
            self.mu = np.array([mu] * dim_process)
        else:
            self.mu = np.array(mu)
            
        if isinstance(alpha, (int, float)):
            self.alpha = np.array([alpha] * dim_process)
        else:
            self.alpha = np.array(alpha)
        
    def simulate(self) -> Tuple[List[np.ndarray]]:
        """
        Génère un processus auto-correctif pour chaque dimension.
        
        Returns:
            Tuple[List[np.ndarray]]: Tuple d'arrays de temps d'événements pour chaque dimension
        """
        events = []
        
        for dim in range(self.dim_process):
            # Simulate each dimension independently
            t = self.start_time
            x = 0
            dim_events = []
            
            while t < self.end_time:
                e = np.random.exponential()
                tau = np.log(e*self.mu[dim]/np.exp(x) + 1)/self.mu[dim]
                t = t + tau
                
                if t >= self.end_time:
                    break
                    
                dim_events.append(t)
                x = x + self.mu[dim]*tau
                x = x - self.alpha[dim]
            
            events.append(np.array(dim_events))
        
        return tuple(events)
    
    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.
        
        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {
            'self_correcting_parameters': {
                'mu': self.mu.tolist(),
                'alpha': self.alpha.tolist()
            }
        }