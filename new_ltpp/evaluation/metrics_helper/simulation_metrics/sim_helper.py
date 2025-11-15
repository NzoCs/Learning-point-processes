"""Simulation metrics computation separated into its own module.

Contains `SimMetricsHelper` which uses extractors from
`.extractor` to compute simulation metrics.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from scipy.stats import wasserstein_distance

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .sim_types import SimMetrics
from .sim_extractor import SimulationDataExtractor


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

        if SimMetrics.WASSERSTEIN_1D.value in self.selected_metrics:
            wasserstein_distance_value = self._wasserstein_1d(
                time_values.true_time_seqs, time_values.sim_time_seqs
            )
            metrics["wasserstein_1d"] = float(wasserstein_distance_value.item())

        if SimMetrics.MMD_RBF_PADDED.value in self.selected_metrics:
            mmd_rbf = self._mmd_rbf(time_values.true_time_seqs, time_values.sim_time_seqs)
            metrics["mmd_rbf_padded"] = float(mmd_rbf.item())

        if SimMetrics.MMD_WASSERSTEIN.value in self.selected_metrics:
            mmd_wasserstein = self._mmd_wasserstein(
                time_values.true_time_seqs, time_values.sim_time_seqs
            )
            metrics["mmd_wasserstein"] = float(mmd_wasserstein.item())

        return metrics

    def compute_all_time_metrics(self, batch: Batch, pred: SimulationResult) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        time_values, _ = self._data_extractor.extract_values(batch, pred)

        wasserstein_distance_value = self._wasserstein_1d(
            time_values.true_time_seqs, time_values.sim_time_seqs
        )
        metrics["wasserstein_1d"] = float(wasserstein_distance_value.item())

        mmd_rbf = self._mmd_rbf(time_values.true_time_seqs, time_values.sim_time_seqs)
        metrics["mmd_rbf_padded"] = float(mmd_rbf.item())

        mmd_wasserstein = self._mmd_wasserstein(
            time_values.true_time_seqs, time_values.sim_time_seqs
        )
        metrics["mmd_wasserstein"] = float(mmd_wasserstein.item())

        return metrics

    def compute_all_type_metrics(self, batch: Batch, pred: SimulationResult) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        _, type_values = self._data_extractor.extract_values(batch, pred)
        type_distance = self._wasserstein_1d(type_values.true_type_seqs, type_values.sim_type_seqs)
        metrics["type_wasserstein"] = float(type_distance.item())
        return metrics

    def get_available_metrics(self) -> List[str]:
        return [
            "wasserstein_1d",
            "mmd_rbf_padded",
            "mmd_wasserstein",
        ]

    # --- internal helpers ---
    def _wasserstein_1d(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_np = true_seqs.cpu().numpy()
        sim_np = sim_seqs.cpu().numpy()
        dist = wasserstein_distance(true_np, sim_np)
        return torch.tensor(dist, device=device)

    def _mmd_rbf(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_vals = true_seqs.view(-1, 1)
        sim_vals = sim_seqs.view(-1, 1)

        def rbf(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _mmd_wasserstein(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        device = true_seqs.device if true_seqs.numel() > 0 else sim_seqs.device
        if true_seqs.numel() == 0 or sim_seqs.numel() == 0:
            return torch.tensor(float("nan"), device=device)

        true_np = true_seqs.cpu().numpy()
        sim_np = sim_seqs.cpu().numpy()
        cross_distance = wasserstein_distance(true_np, sim_np)
        k_ts = torch.exp(-torch.tensor(cross_distance**2, device=device) / (2 * sigma**2))
        return 2 - 2 * k_ts

