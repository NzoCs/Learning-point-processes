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

class MultivariableSelfCorrecting(BaseSimulator):
    """
    Simulateur de processus ponctuels temporels avec correction automatique multivariable.
    Cette version permet aux événements de différentes dimensions d'avoir une influence les uns sur les autres.
    """
    
    def __init__(self, 
                 dim_process: int,
                 mu: float = 1.0,
                 alpha_matrix: Optional[np.ndarray] = None,
                 start_time: float = 100,
                 end_time: float = 200,
                 seed: Optional[int] = None):
        """
        Initialise un simulateur de processus ponctuel temporel avec correction automatique multivariable.
        
        Args:
            dim_process (int): Dimension du processus (nombre de types d'événements)
            mu (float): Paramètre de taux de base (scalaire ou array)
            alpha_matrix (np.ndarray, optional): Matrice d'influence entre dimensions. 
                                               alpha_matrix[i,j] indique l'influence d'un événement de type j sur le taux de la dimension i.
                                               Si None, une matrice identité est utilisée (pas d'influence croisée).
            start_time (float): Temps de début de la simulation
            end_time (float): Temps de fin de la simulation
            seed (int, optional): Graine pour la reproductibilité
        """
        super().__init__(dim_process, start_time, end_time, seed)
        
        # Support both scalar and array inputs for mu
        if isinstance(mu, (int, float)):
            self.mu = np.array([mu] * dim_process)
        else:
            self.mu = np.array(mu)
        
        # Initialize alpha_matrix
        if alpha_matrix is None:
            # Default: identity matrix (no cross-influence)
            self.alpha_matrix = np.eye(dim_process)
        else:
            if alpha_matrix.shape != (dim_process, dim_process):
                raise ValueError(f"alpha_matrix must be of shape ({dim_process}, {dim_process})")
            self.alpha_matrix = alpha_matrix
        
    def simulate(self) -> Tuple[List[np.ndarray]]:
        """
        Génère un processus auto-correctif multivariable.
        
        Returns:
            Tuple[List[np.ndarray]]: Tuple d'arrays de temps d'événements pour chaque dimension
        """
        # Initialize state variables for each dimension
        x = np.zeros(self.dim_process)
        t = self.start_time
        next_event_times = np.full(self.dim_process, np.inf)
        events_by_dim = [[] for _ in range(self.dim_process)]
        
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
            events_by_dim[next_dim].append(next_time)
            
            # Apply the influence of the event on all dimensions
            x -= self.alpha_matrix[:, next_dim]
            
            # Generate next event time for the dimension that just had an event
            e = np.random.exponential()
            tau = np.log(e * self.mu[next_dim] / np.exp(x[next_dim]) + 1) / self.mu[next_dim]
            next_event_times[next_dim] = next_time + tau
            
            # Update current time
            t = next_time
        
        # Convert event lists to numpy arrays
        return tuple(np.array(events) for events in events_by_dim)
    
    def get_simulator_metadata(self) -> Dict:
        """
        Renvoie les métadonnées spécifiques au simulateur.
        
        Returns:
            Dict: Métadonnées spécifiques au simulateur
        """
        return {
            'multivariable_self_correcting_parameters': {
                'mu': self.mu.tolist(),
                'alpha_matrix': self.alpha_matrix.tolist()
            }
        }