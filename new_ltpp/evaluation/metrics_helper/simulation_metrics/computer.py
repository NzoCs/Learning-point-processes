"""Simulation metrics computation separated into its own module.

Contains `SimulationMetricsComputer` which uses extractors from
`.extractor` to compute simulation metrics.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .simul_types import SimulationMetrics, SimulationTimeValues, SimulationTimeValues
from .extractor import SimulationDataExtractor, SimulationTimeDataExtractor, SimulationTypeDataExtractor


class SimulationMetricsComputer():
    """Computes simulation-specific metrics."""

    def __init__(self, num_event_types: int, selected_metrics: Optional[List[Union[str, SimulationMetrics]]] = None):
        self.num_event_types = num_event_types
        self._data_extractor = SimulationDataExtractor(num_event_types)
        self._time_extractor = SimulationTimeDataExtractor(num_event_types)
        self._type_extractor = SimulationTypeDataExtractor(num_event_types)

        if selected_metrics is None:
            self.selected_metrics = set(self.get_available_metrics())
        else:
            processed_metrics = []
            for metric in selected_metrics:
                if isinstance(metric, SimulationMetrics):
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
        try:
            (
                true_time_seqs,
                true_type_seqs,
                true_time_delta_seqs,
                sim_time_seqs,
                sim_type_seqs,
                sim_mask,
            ) = self._data_extractor.extract_values(batch, pred)

            metrics: Dict[str, float] = {}

            if SimulationMetrics.WASSERSTEIN_1D.value in self.selected_metrics:
                wasserstein_distances = self._batch_wasserstein_1d(true_time_seqs, sim_time_seqs, sim_mask)
                metrics["wasserstein_1d"] = float(wasserstein_distances.mean().item())

            if SimulationMetrics.MMD_RBF_PADDED.value in self.selected_metrics:
                mmd_rbf = self._batch_mmd_rbf_padded(true_time_seqs, sim_time_seqs, sim_mask)
                metrics["mmd_rbf_padded"] = float(mmd_rbf.item())

            if SimulationMetrics.MMD_WASSERSTEIN.value in self.selected_metrics:
                mmd_wasserstein = self._batch_mmd_wasserstein(true_time_seqs, sim_time_seqs, sim_mask)
                metrics["mmd_wasserstein"] = float(mmd_wasserstein.item())

            return metrics

        except Exception as e:
            logger.error(f"Error computing simulation metrics: {e}")
            return self._get_nan_metrics()

    def compute_all_time_metrics(self, batch: Batch, pred: SimulationResult) -> Dict[str, Any]:
        try:
            metrics: Dict[str, Any] = {}
            time_values = self._time_extractor.extract_simulation_time_values(batch, pred)

            wasserstein_distances = self._batch_wasserstein_1d_from_time_values(time_values)
            metrics["wasserstein_1d"] = float(wasserstein_distances.mean().item())

            mmd_rbf = self._batch_mmd_rbf_padded_from_time_values(time_values)
            metrics["mmd_rbf_padded"] = float(mmd_rbf.item())

            mmd_wasserstein = self._batch_mmd_wasserstein_from_time_values(time_values)
            metrics["mmd_wasserstein"] = float(mmd_wasserstein.item())

            return metrics

        except Exception as e:
            logger.error(f"Error computing simulation time metrics: {e}")
            return {
                "wasserstein_1d": float("nan"),
                "mmd_rbf_padded": float("nan"),
                "mmd_wasserstein": float("nan"),
            }

    def compute_all_type_metrics(self, batch: Batch, pred: SimulationResult) -> Dict[str, Any]:
        try:
            metrics: Dict[str, Any] = {}
            return metrics
        except Exception as e:
            logger.error(f"Error computing simulation type metrics: {e}")
            return {}

    def get_available_metrics(self) -> List[str]:
        return ["wasserstein_1d", "mmd_rbf_padded", "mmd_wasserstein"]

    # --- internal helpers (copied from previous module) ---
    def _batch_wasserstein_1d(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = true_seqs.shape
        true_sorted = torch.sort(torch.where(mask, true_seqs, float("inf")), dim=1).values
        sim_sorted = torch.sort(torch.where(mask, sim_seqs, float("inf")), dim=1).values
        lengths = mask.sum(dim=1)
        dists = torch.zeros(batch_size, device=true_seqs.device)

        for i in range(batch_size):
            n = lengths[i].item()
            if n == 0:
                dists[i] = float("inf")
            else:
                x = true_sorted[i, :n].cpu().numpy()
                y = sim_sorted[i, :n].cpu().numpy()
                dists[i] = wasserstein_distance(x, y)

        return dists

    def _batch_mmd_rbf_padded(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        batch_size, seq_len = true_seqs.shape
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item())
        if max_len == 0:
            return torch.tensor(float("nan"), device=true_seqs.device)

        true_pad = torch.zeros(batch_size, max_len, device=true_seqs.device)
        sim_pad = torch.zeros(batch_size, max_len, device=sim_seqs.device)

        for i in range(batch_size):
            n = lengths[i].item()
            if n > 0:
                valid_true = true_seqs[i][mask[i]][:n]
                valid_sim = sim_seqs[i][mask[i]][:n]
                true_pad[i, : len(valid_true)] = valid_true
                sim_pad[i, : len(valid_sim)] = valid_sim

        def rbf(X, Y):
            XX = torch.cdist(X, X) ** 2
            YY = torch.cdist(Y, Y) ** 2
            XY = torch.cdist(X, Y) ** 2
            return (
                torch.exp(-XX / (2 * sigma**2)).mean(),
                torch.exp(-YY / (2 * sigma**2)).mean(),
                torch.exp(-XY / (2 * sigma**2)).mean(),
            )

        kxx, kyy, kxy = rbf(true_pad, sim_pad)
        return kxx + kyy - 2 * kxy

    def _batch_mmd_wasserstein(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        batch_size = true_seqs.size(0)
        true_list = []
        sim_list = []

        for i in range(batch_size):
            valid_mask = mask[i].bool()
            true_list.append(true_seqs[i][valid_mask].cpu().numpy())
            sim_list.append(sim_seqs[i][valid_mask].cpu().numpy())

        W_tt = np.zeros((batch_size, batch_size))
        W_ss = np.zeros((batch_size, batch_size))
        W_ts = np.zeros((batch_size, batch_size))

        for i in range(batch_size):
            for j in range(batch_size):
                W_tt[i, j] = wasserstein_distance(true_list[i], true_list[j])
                W_ss[i, j] = wasserstein_distance(sim_list[i], sim_list[j])
                W_ts[i, j] = wasserstein_distance(true_list[i], sim_list[j])

        T = torch.tensor(W_tt, device=true_seqs.device)
        S = torch.tensor(W_ss, device=true_seqs.device)
        X = torch.tensor(W_ts, device=true_seqs.device)

        K_tt = torch.exp(-(T**2) / (2 * sigma**2)).mean()
        K_ss = torch.exp(-(S**2) / (2 * sigma**2)).mean()
        K_ts = torch.exp(-(X**2) / (2 * sigma**2)).mean()

        return K_tt + K_ss - 2 * K_ts

    def _batch_wasserstein_1d_from_time_values(self, time_values: SimulationTimeValues):
        return self._batch_wasserstein_1d(time_values.true_time_seqs, time_values.sim_time_seqs, time_values.sim_mask)

    def _batch_mmd_rbf_padded_from_time_values(self, time_values: SimulationTimeValues, sigma: float = 1.0):
        return self._batch_mmd_rbf_padded(time_values.true_time_seqs, time_values.sim_time_seqs, time_values.sim_mask, sigma)

    def _batch_mmd_wasserstein_from_time_values(self, time_values: SimulationTimeValues, sigma: float = 1.0):
        return self._batch_mmd_wasserstein(time_values.true_time_seqs, time_values.sim_time_seqs, time_values.sim_mask, sigma)

    def _get_nan_metrics(self) -> Dict[str, float]:
        return {
            "wasserstein_1d": float("nan"),
            "mmd_rbf_padded": float("nan"),
            "mmd_wasserstein": float("nan"),
        }
