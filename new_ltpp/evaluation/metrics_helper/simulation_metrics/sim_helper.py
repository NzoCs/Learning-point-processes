"""Simulation metrics computation separated into its own module.

Contains `SimMetricsHelper` which uses extractors from
`.extractor` to compute simulation metrics.
"""

from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from scipy.stats import wasserstein_distance

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .sim_extractor import SimulationDataExtractor
from .sim_types import SimMetrics, SimTimeValues


class SimMetricsHelper:
    """Computes simulation-specific metrics."""

    def __init__(
        self,
        num_event_types: int,
        selected_metrics: Optional[List[Union[str, SimMetrics]]] = None,
    ):
        self.num_event_types = num_event_types
        self._data_extractor = SimulationDataExtractor(num_event_types)

        if selected_metrics is None:
            self.selected_metrics = set(self.get_available_metrics())
        else:
            processed_metrics: List[str] = []
            for metric in selected_metrics:
                if isinstance(metric, SimMetrics):
                    processed_metrics.append(metric.value)
                else:
                    processed_metrics.append(str(metric))

            available = set(self.get_available_metrics())
            selected_set = set(processed_metrics)
            invalid_metrics = selected_set - available
            if invalid_metrics:
                logger.warning(
                    f"Invalid simulation metrics requested: {invalid_metrics}. Available: {available}"
                )
                selected_set = selected_set & available

            self.selected_metrics = selected_set

    def compute_metrics(self, batch: Batch, pred: SimulationResult) -> Dict[str, float]:
        time_values, _ = self._data_extractor.extract_values(batch, pred)

        metrics: Dict[str, float] = {}
        metric_mapping = self._build_time_metric_mapping(time_values)

        for metric_name, (func, *args) in metric_mapping.items():
            if metric_name in self.selected_metrics:
                metrics[metric_name] = float(func(*args).item())

        return metrics

    def compute_all_time_metrics(
        self, batch: Batch, pred: SimulationResult
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        time_values, _ = self._data_extractor.extract_values(batch, pred)

        metric_mapping = self._build_time_metric_mapping(time_values)
        for metric_name, (func, *args) in metric_mapping.items():
            metrics[metric_name] = float(func(*args).item())

        return metrics

    def compute_all_type_metrics(
        self, batch: Batch, pred: SimulationResult
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        _, type_values = self._data_extractor.extract_values(batch, pred)
        type_distance = self._wasserstein_1d(
            type_values["true_type_seqs"], type_values["sim_type_seqs"]
        )
        metrics["type_wasserstein"] = float(type_distance.item())
        return metrics

    def get_available_metrics(self) -> List[str]:
        return [
            "wasserstein_1d",
            "mmd_rbf_padded",
            "mmd_wasserstein",
        ]

    # --- internal helpers ---
    def _wasserstein_1d(
        self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor
    ) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_np = true_seqs.cpu().numpy()
        sim_np = sim_seqs.cpu().numpy()
        dist = wasserstein_distance(true_np, sim_np)
        return torch.tensor(dist, device=device)

    def _mmd_rbf(
        self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_vals = true_seqs.view(-1, 1)
        sim_vals = sim_seqs.view(-1, 1)

        def rbf(
            X: torch.Tensor, Y: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            XX = torch.cdist(X, X) ** 2
            YY = torch.cdist(Y, Y) ** 2
            XY = torch.cdist(X, Y) ** 2
            return (
                torch.exp(-XX / (2 * sigma**2)).mean(),
                torch.exp(-YY / (2 * sigma**2)).mean(),
                torch.exp(-XY / (2 * sigma**2)).mean(),
            )

        kxx, kyy, kxy = rbf(true_vals, sim_vals)
        return kxx + kyy - 2 * kxy

    def _mmd_wasserstein(
        self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, sigma: float = 1.0
    ) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_np = true_seqs.cpu().numpy()
        sim_np = sim_seqs.cpu().numpy()
        cross_distance = wasserstein_distance(true_np, sim_np)
        k_ts = torch.exp(
            -torch.tensor(cross_distance**2, device=device) / (2 * sigma**2)
        )
        return 2 - 2 * k_ts

    def _build_time_metric_mapping(
        self, time_values: SimTimeValues
    ) -> Dict[str, tuple[Callable[..., torch.Tensor], Any, Any]]:
        return {
            SimMetrics.WASSERSTEIN_1D.value: (
                self._wasserstein_1d,
                time_values["true_time_seqs"],
                time_values["sim_time_seqs"],
            ),
            SimMetrics.MMD_RBF_PADDED.value: (
                self._mmd_rbf,
                time_values["true_time_seqs"],
                time_values["sim_time_seqs"],
            ),
            SimMetrics.MMD_WASSERSTEIN.value: (
                self._mmd_wasserstein,
                time_values["true_time_seqs"],
                time_values["sim_time_seqs"],
            ),
        }

# """Fixed simulation metrics with proper handling of padding and batches."""

# import torch
# from scipy.stats import wasserstein_distance
# from typing import Dict, Any, Tuple


# class SimMetricsHelperFixed:
#     """Improved metrics computation with proper padding handling."""

#     def _extract_valid_values(
#         self, 
#         seqs: torch.Tensor, 
#         mask: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Extract only valid (non-padded) values from sequences.
        
#         Args:
#             seqs: (B, L) tensor of values
#             mask: (B, L) boolean mask (True = valid)
            
#         Returns:
#             1D tensor of all valid values concatenated
#         """
#         return seqs[mask]
    
#     def _wasserstein_1d(
#         self, 
#         true_seqs: torch.Tensor, 
#         sim_seqs: torch.Tensor,
#         true_mask: torch.Tensor,
#         sim_mask: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute Wasserstein distance between valid values only.
        
#         Args:
#             true_seqs: (B, L) true sequences
#             sim_seqs: (B, L) simulated sequences
#             true_mask: (B, L) validity mask for true sequences
#             sim_mask: (B, L) validity mask for simulated sequences
#         """
#         device = true_seqs.device
        
#         # Extract only valid values
#         true_valid = self._extract_valid_values(true_seqs, true_mask)
#         sim_valid = self._extract_valid_values(sim_seqs, sim_mask)
        
#         if true_valid.numel() == 0 or sim_valid.numel() == 0:
#             return torch.tensor(float("nan"), device=device)
        
#         # Compute Wasserstein on valid values only
#         true_np = true_valid.cpu().numpy()
#         sim_np = sim_valid.cpu().numpy()
#         dist = wasserstein_distance(true_np, sim_np)
        
#         return torch.tensor(dist, device=device)
    
#     def _wasserstein_per_sequence(
#         self,
#         true_seqs: torch.Tensor,
#         sim_seqs: torch.Tensor, 
#         true_mask: torch.Tensor,
#         sim_mask: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute average Wasserstein distance per sequence pair.
#         Better for preserving sequence structure.
        
#         Returns:
#             Average distance across all sequence pairs
#         """
#         B = true_seqs.shape[0]
#         device = true_seqs.device
#         distances = []
        
#         for b in range(B):
#             true_valid = true_seqs[b][true_mask[b]]
#             sim_valid = sim_seqs[b][sim_mask[b]]
            
#             if true_valid.numel() == 0 or sim_valid.numel() == 0:
#                 continue
                
#             true_np = true_valid.cpu().numpy()
#             sim_np = sim_valid.cpu().numpy()
#             dist = wasserstein_distance(true_np, sim_np)
#             distances.append(dist)
        
#         if not distances:
#             return torch.tensor(float("nan"), device=device)
        
#         return torch.tensor(sum(distances) / len(distances), device=device)

#     def _mmd_rbf(
#         self, 
#         true_seqs: torch.Tensor, 
#         sim_seqs: torch.Tensor,
#         true_mask: torch.Tensor,
#         sim_mask: torch.Tensor,
#         sigma: float = 1.0
#     ) -> torch.Tensor:
#         """
#         Compute MMD with RBF kernel, excluding padded values.
        
#         MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
#         where k is the RBF kernel: k(x,y) = exp(-||x-y||²/(2σ²))
#         """
#         device = true_seqs.device
        
#         # Extract valid values only
#         true_valid = self._extract_valid_values(true_seqs, true_mask).view(-1, 1)
#         sim_valid = self._extract_valid_values(sim_seqs, sim_mask).view(-1, 1)
        
#         if true_valid.numel() == 0 or sim_valid.numel() == 0:
#             return torch.tensor(float("nan"), device=device)
        
#         def rbf_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#             """Compute RBF kernel matrix."""
#             dists = torch.cdist(X, Y) ** 2
#             return torch.exp(-dists / (2 * sigma ** 2))
        
#         # Compute kernel matrices
#         K_XX = rbf_kernel(true_valid, true_valid)
#         K_YY = rbf_kernel(sim_valid, sim_valid)
#         K_XY = rbf_kernel(true_valid, sim_valid)
        
#         # MMD² estimator (unbiased)
#         n = K_XX.shape[0]
#         m = K_YY.shape[0]
        
#         # E[k(X,X')] - remove diagonal for unbiased estimator
#         term1 = (K_XX.sum() - K_XX.diagonal().sum()) / (n * (n - 1)) if n > 1 else K_XX.mean()
        
#         # E[k(Y,Y')]
#         term2 = (K_YY.sum() - K_YY.diagonal().sum()) / (m * (m - 1)) if m > 1 else K_YY.mean()
        
#         # E[k(X,Y)]
#         term3 = K_XY.mean()
        
#         mmd_squared = term1 + term2 - 2 * term3
        
#         # Return MMD (not squared), clamped to avoid negative due to numerical errors
#         return torch.sqrt(torch.clamp(mmd_squared, min=0.0))

#     def _mmd_wasserstein_corrected(
#         self,
#         true_seqs: torch.Tensor,
#         sim_seqs: torch.Tensor,
#         true_mask: torch.Tensor,
#         sim_mask: torch.Tensor,
#         sigma: float = 1.0
#     ) -> torch.Tensor:
#         """
#         Compute MMD using Wasserstein distance as the base metric in RBF kernel.
        
#         This uses: k(X,Y) = exp(-W(X,Y)²/(2σ²))
#         where W(X,Y) is the Wasserstein distance.
        
#         Note: This requires computing pairwise Wasserstein distances which is expensive.
#         For large datasets, consider using standard RBF MMD instead.
#         """
#         device = true_seqs.device
        
#         # For this metric, we compute Wasserstein-based kernel
#         # which is computationally expensive
#         true_valid = self._extract_valid_values(true_seqs, true_mask)
#         sim_valid = self._extract_valid_values(sim_seqs, sim_mask)
        
#         if true_valid.numel() == 0 or sim_valid.numel() == 0:
#             return torch.tensor(float("nan"), device=device)
        
#         # Simplified version: just use Wasserstein distance directly
#         # (Not a true MMD but more interpretable)
#         true_np = true_valid.cpu().numpy()
#         sim_np = sim_valid.cpu().numpy()
#         w_dist = wasserstein_distance(true_np, sim_np)
        
#         return torch.tensor(w_dist, device=device)


# # Usage example showing the difference:
# def compare_metrics_example():
#     """Demonstrate the difference between old and new implementations."""
    
#     # Example data with padding
#     true_seqs = torch.tensor([
#         [1.0, 2.0, 3.0, 0.0, 0.0],  # Last 2 are padding
#         [1.5, 2.5, 0.0, 0.0, 0.0],  # Last 3 are padding
#     ])
    
#     sim_seqs = torch.tensor([
#         [1.1, 2.1, 3.1, 0.0, 0.0],
#         [1.4, 2.6, 0.0, 0.0, 0.0],
#     ])
    
#     true_mask = torch.tensor([
#         [True, True, True, False, False],
#         [True, True, False, False, False],
#     ])
    
#     sim_mask = torch.tensor([
#         [True, True, True, False, False],
#         [True, True, False, False, False],
#     ])
    
#     helper = SimMetricsHelperFixed()
    
#     # OLD WAY (incorrect - includes padding):
#     # true_np = true_seqs.cpu().numpy()  # [1,2,3,0,0,1.5,2.5,0,0,0]
#     # sim_np = sim_seqs.cpu().numpy()    # [1.1,2.1,3.1,0,0,1.4,2.6,0,0,0]
#     # Would compare including all the 0s!
    
#     # NEW WAY (correct - excludes padding):
#     w_dist = helper._wasserstein_1d(true_seqs, sim_seqs, true_mask, sim_mask)
#     # Only compares: [1,2,3,1.5,2.5] vs [1.1,2.1,3.1,1.4,2.6]
    
#     print(f"Wasserstein distance (valid only): {w_dist.item():.4f}")
    
#     # Per-sequence alternative:
#     w_per_seq = helper._wasserstein_per_sequence(true_seqs, sim_seqs, true_mask, sim_mask)
#     print(f"Wasserstein per sequence: {w_per_seq.item():.4f}")
    
#     # MMD with proper padding handling:
#     mmd = helper._mmd_rbf(true_seqs, sim_seqs, true_mask, sim_mask, sigma=1.0)
#     print(f"MMD (valid only): {mmd.item():.4f}")


# if __name__ == "__main__":
#     compare_metrics_example()