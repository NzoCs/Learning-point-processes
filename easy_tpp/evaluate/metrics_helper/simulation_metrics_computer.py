"""
Simulation metrics computation class.

This class is responsible solely for computing simulation-related metrics,
following the Single Responsibility Principle.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import wasserstein_distance
from .interfaces import MetricsComputerInterface, DataExtractorInterface
from easy_tpp.utils import logger


class SimulationDataExtractor(DataExtractorInterface):
    """Extracts simulation data from batch and predictions."""
    
    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
    
    def extract_values(self, batch: Any, pred: Any) -> Tuple[torch.Tensor, ...]:
        """Extract simulation values for metrics computation."""
        if len(batch) >= 4:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask = batch[:4]
        else:
            true_time_seqs, true_time_delta_seqs, true_type_seqs = batch[:3]
            batch_non_pad_mask = (true_type_seqs != self.num_event_types).float()
        
        sim_time_seqs, sim_time_delta_seqs, sim_type_seqs = pred[:3]
        sim_mask = pred[3] if len(pred) >= 4 else torch.ones_like(sim_type_seqs, dtype=torch.bool)
        
        return true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask


class SimulationMetricsComputer(MetricsComputerInterface):
    """
    Computes simulation-specific metrics.
    
    This class focuses solely on simulation metrics computation,
    adhering to the Single Responsibility Principle.
    """
    
    def __init__(self, num_event_types: int, data_extractor: DataExtractorInterface = None):
        """
        Initialize the simulation metrics computer.
        
        Args:
            num_event_types: Number of event types
            data_extractor: Custom data extractor (optional)
        """
        self.num_event_types = num_event_types
        self._data_extractor = data_extractor or SimulationDataExtractor(num_event_types)
    
    def compute_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """
        Compute all simulation metrics.
        
        Args:
            batch: Input batch data
            pred: Model predictions
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            extracted_values = self._data_extractor.extract_values(batch, pred)
            true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask = extracted_values
            
            metrics = {}
            
            # Calculate Wasserstein 1D distance per sequence
            wasserstein_distances = self._batch_wasserstein_1d(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['wasserstein_1d'] = float(wasserstein_distances.mean().item())

            # Calculate MMD RBF with padding
            mmd_rbf = self._batch_mmd_rbf_padded(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['mmd_rbf_padded'] = float(mmd_rbf.item())

            # Calculate MMD with Wasserstein kernel
            mmd_wasserstein = self._batch_mmd_wasserstein(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['mmd_wasserstein'] = float(mmd_wasserstein.item())

            return metrics
            
        except Exception as e:
            logger.error(f"Error computing simulation metrics: {e}")
            return self._get_nan_metrics()
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available simulation metrics."""
        return [
            'wasserstein_1d',
            'mmd_rbf_padded',
            'mmd_wasserstein'
        ]
    
    def _batch_wasserstein_1d(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate Wasserstein 1D distance for each sequence pair in the batch.
        
        Args:
            true_seqs: [batch_size, seq_len] true temporal sequences
            sim_seqs: [batch_size, seq_len] simulated sequences
            mask: [batch_size, seq_len] boolean mask
            
        Returns:
            torch.Tensor: [batch_size] Wasserstein distances
        """
        batch_size, seq_len = true_seqs.shape
        # Apply mask and sort
        true_sorted = torch.sort(torch.where(mask, true_seqs, float('inf')), dim=1).values
        sim_sorted = torch.sort(torch.where(mask, sim_seqs, float('inf')), dim=1).values
        lengths = mask.sum(dim=1)
        dists = torch.zeros(batch_size, device=true_seqs.device)

        for i in range(batch_size):
            n = lengths[i].item()
            if n == 0:
                dists[i] = float('inf')
            else:
                x = true_sorted[i, :n].cpu().numpy()
                y = sim_sorted[i, :n].cpu().numpy()
                dists[i] = wasserstein_distance(x, y)
                
        return dists

    def _batch_mmd_rbf_padded(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Biased MMD^2 with RBF kernel on padded sequences.
        
        Args:
            true_seqs: [batch_size, seq_len] true temporal sequences
            sim_seqs: [batch_size, seq_len] simulated sequences
            mask: [batch_size, seq_len] boolean mask
            sigma: RBF bandwidth parameter
            
        Returns:
            torch.Tensor: MMD squared value
        """
        batch_size, seq_len = true_seqs.shape
        # Calculate lengths and max
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item())
        if max_len == 0:
            return torch.tensor(float('nan'), device=true_seqs.device)

        # Prepare padding
        true_pad = torch.zeros(batch_size, max_len, device=true_seqs.device)
        sim_pad = torch.zeros(batch_size, max_len, device=sim_seqs.device)

        for i in range(batch_size):
            n = lengths[i].item()
            if n > 0:
                valid_true = true_seqs[i][mask[i]][:n]
                valid_sim = sim_seqs[i][mask[i]][:n]
                true_pad[i, :len(valid_true)] = valid_true
                sim_pad[i, :len(valid_sim)] = valid_sim

        # Calculate kernel matrices
        def rbf(X, Y):
            # X: [batch, d], Y: [batch, d]
            XX = torch.cdist(X, X)**2
            YY = torch.cdist(Y, Y)**2
            XY = torch.cdist(X, Y)**2
            return torch.exp(-XX/(2*sigma**2)).mean(), \
                   torch.exp(-YY/(2*sigma**2)).mean(), \
                   torch.exp(-XY/(2*sigma**2)).mean()

        kxx, kyy, kxy = rbf(true_pad, sim_pad)
        return kxx + kyy - 2*kxy

    def _batch_mmd_wasserstein(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Biased MMD^2 with RBF kernel defined on Wasserstein distances.
        
        Args:
            true_seqs: [batch_size, seq_len] true temporal sequences
            sim_seqs: [batch_size, seq_len] simulated sequences
            mask: [batch_size, seq_len] boolean mask
            sigma: RBF bandwidth parameter
            
        Returns:
            torch.Tensor: MMD squared value
        """
        batch_size = true_seqs.size(0)
        # Extract valid subsequences
        true_list = []
        sim_list = []
        
        for i in range(batch_size):
            valid_mask = mask[i].bool()
            true_list.append(true_seqs[i][valid_mask].cpu().numpy())
            sim_list.append(sim_seqs[i][valid_mask].cpu().numpy())

        # Calculate Wasserstein distance matrices
        W_tt = np.zeros((batch_size, batch_size))
        W_ss = np.zeros((batch_size, batch_size))
        W_ts = np.zeros((batch_size, batch_size))
        
        for i in range(batch_size):
            for j in range(batch_size):
                W_tt[i,j] = wasserstein_distance(true_list[i], true_list[j])
                W_ss[i,j] = wasserstein_distance(sim_list[i], sim_list[j])
                W_ts[i,j] = wasserstein_distance(true_list[i], sim_list[j])

        # Convert to torch and kernel
        T = torch.tensor(W_tt, device=true_seqs.device)
        S = torch.tensor(W_ss, device=true_seqs.device)
        X = torch.tensor(W_ts, device=true_seqs.device)

        K_tt = torch.exp(-T**2/(2*sigma**2)).mean()
        K_ss = torch.exp(-S**2/(2*sigma**2)).mean()
        K_ts = torch.exp(-X**2/(2*sigma**2)).mean()

        return K_tt + K_ss - 2*K_ts
    
    def _get_nan_metrics(self) -> Dict[str, float]:
        """Get a dictionary of NaN metrics for error cases."""
        return {
            'wasserstein_1d': float('nan'),
            'mmd_rbf_padded': float('nan'),
            'mmd_wasserstein': float('nan')
        }
