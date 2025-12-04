"""Fixed simulation metrics with proper handling of padding and batches."""

import torch
from scipy.stats import wasserstein_distance
from typing import Dict, Any, Tuple, List

from new_ltpp.shared_types import Batch, SimulationResult


class SimMetricsHelperFixed:
    """Improved metrics computation with proper padding handling."""

    def _extract_valid_values(
        self,
        seqs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract only valid (non-padded) values from sequences.

        Args:
            seqs: (B, L) tensor of values
            mask: (B, L) boolean mask (True = valid)

        Returns:
            1D tensor of all valid values concatenated
        """
        return seqs[mask]
    
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
            type_values["true_type_seqs"], type_values["sim_type_seqs"],
            type_values["true_type_mask"], type_values["sim_type_mask"]
        )
        metrics["type_wasserstein"] = float(type_distance.item())
        return metrics


    def get_available_metrics(self) -> List[str]:
        return [
            "wasserstein_1d",
            "mmd_rbf_padded",
            "mmd_wasserstein",
        ]


    def _wasserstein_1d(
        self,
        true_seqs: torch.Tensor,
        sim_seqs: torch.Tensor,
        true_mask: torch.Tensor,
        sim_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wasserstein distance between valid values only.

        Args:
            true_seqs: (B, L) true sequences
            sim_seqs: (B, L) simulated sequences
            true_mask: (B, L) validity mask for true sequences
            sim_mask: (B, L) validity mask for simulated sequences
        """

        device = true_seqs.device
*
        if .numel() == 0 or sim_valid.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        # Compute Wasserstein on valid values only
        true_np = true_valid.cpu().numpy()
        sim_np = sim_valid.cpu().numpy()
        dist = wasserstein_distance(true_np, sim_np)

        return torch.tensor(dist, device=device)
