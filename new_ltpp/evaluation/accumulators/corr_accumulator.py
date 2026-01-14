"""
Autocorrelation Accumulator

Accumulates autocorrelation statistics for temporal point processes.
"""

import numpy as np
import torch
import torch.fft

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .acc_types import CorrelationStatistics
from .base_accumulator import BaseAccumulator


class CorrAccumulator(BaseAccumulator):
    """Accumulates autocorrelation statistics for TPP sequences."""

    def __init__(self, min_sim_events: int = 1, nb_bins: int = 10, max_lag: int = 15):
        super().__init__(min_sim_events)
        self.nb_bins = nb_bins
        self.max_lag = max_lag
        self.acf_gt_mean = np.zeros(max_lag + 1)
        self.acf_sim_mean = np.zeros(max_lag + 1)
        self.batch_count = 0

    def update(self, batch: Batch, simulation: SimulationResult) -> None:
        """Update accumulator with ACF from batch and simulation.

        Args:
            batch: Ground truth batch
            simulation: Simulation result (required)
        """

        # Validate simulation has sufficient events
        if simulation is None:
            logger.warning(
                "CorrAccumulator: No simulation results provided, skipping batch"
            )
            return

        # Compute ACF for ground truth
        acf_gt = self.compute_acf_from_batch(batch, self.nb_bins, self.max_lag)
        acf_gt_np = acf_gt.cpu().numpy()  # (B, max_lag+1)
        acf_gt_batch_mean = np.mean(acf_gt_np, axis=0)  # (max_lag+1,)

        # Compute ACF for simulation
        acf_sim = self.compute_acf_from_simulation(
            simulation, self.nb_bins, self.max_lag
        )
        acf_sim_np = acf_sim.cpu().numpy()  # (B, max_lag+1)
        acf_sim_batch_mean = np.mean(acf_sim_np, axis=0)  # (max_lag+1,)

        # Update running mean
        self.batch_count += 1
        n = self.batch_count
        self.acf_gt_mean = ((n - 1) * self.acf_gt_mean + acf_gt_batch_mean) / n
        self.acf_sim_mean = ((n - 1) * self.acf_sim_mean + acf_sim_batch_mean) / n

        self._sample_count += batch.time_seqs.shape[0]

    def compute(self) -> CorrelationStatistics:
        """Compute final autocorrelation statistics.

        Returns:
            CorrelationStatistics with mean ACF for ground truth and simulation
        """
        if self.batch_count == 0:
            logger.warning("No ACF data accumulated")
            return CorrelationStatistics(
                acf_gt_mean=np.zeros(self.max_lag + 1),
                acf_sim_mean=np.zeros(self.max_lag + 1),
            )

        return CorrelationStatistics(
            acf_gt_mean=self.acf_gt_mean,
            acf_sim_mean=self.acf_sim_mean,
        )

    @staticmethod
    def create_hist(
        times: torch.Tensor,
        mask: torch.Tensor,
        nb_bins: int,
    ) -> torch.Tensor:
        """
        Create histogram of binned counts for a single sequence.

        Args:
            times: Tensor of shape (B, L,) with event times
            mask: Tensor of shape (B, L,) with 1 for valid events, 0 for padding
            nb_bins: number of bins for discretization
        Returns:
            Tensor of shape (B, nb_bins) with binned counts
        """

        B = times.shape[0]

        # Max time per sequence
        max_times = torch.max(times * mask, dim=1).values  # (B,)
        max_time_global = torch.max(max_times).item()

        # Calculate bin_width based on nb_bins
        bin_width = max_time_global / nb_bins if max_time_global > 0 else 1.0

        # Compute bin indices for all times: (B, L)
        bin_indices = (times / bin_width).long()  # (B, L)
        bin_indices = torch.where(
            mask, bin_indices, nb_bins
        )  # Set masked positions to nb_bins (out of range)
        bin_indices = torch.clamp(bin_indices, 0, nb_bins - 1)

        # Zero histogram
        hist = torch.zeros((B, nb_bins + 1), dtype=torch.float, device=times.device)

        # Use scatter_add to accumulate counts per batch
        # For each valid time, add 1 to the corresponding bin
        hist.scatter_add_(
            dim=1,
            index=bin_indices,
            src=mask.float(),  # Add 1 where mask is True, 0 otherwise
        )

        hist = hist[:, :nb_bins]  # Discard last bin used for padding

        return hist

    @staticmethod
    def compute_acf_from_batch(
        batch: Batch,
        nb_bins: int,
        max_lag: int,
    ) -> torch.Tensor:
        """
        Compute ACF of binned counts for each sequence in the batch using FFT.
        Uses scatter_add for efficient histogram computation per sequence.

        Args:
            batch: Batch containing time_seqs and seq_non_pad_mask
            nb_bins: number of bins for discretization
            max_lag: maximum integer lag to compute ACF

        Returns:
            Tensor of shape (batch_size, max_lag + 1)
        """

        # (B, nb_bins)
        hist = CorrAccumulator.create_hist(
            times=batch.time_seqs,
            mask=batch.valid_event_mask,
            nb_bins=nb_bins,
        )  # (B, nb_bins)

        # Mean-center
        hist_centered = hist - hist.mean(dim=1, keepdim=True)
        denom = torch.sum(hist_centered**2, dim=1)

        # Compute ACF using FFT
        n_fft = 2 ** (nb_bins - 1).bit_length() * 2

        fft_hist = torch.fft.rfft(hist_centered, n=n_fft, dim=1)
        power_spectrum = fft_hist * torch.conj(fft_hist)
        acf_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=1)

        acf = acf_full[:, : max_lag + 1] / denom.unsqueeze(1)

        return acf

    @staticmethod
    def compute_acf_from_simulation(
        sim: SimulationResult,
        nb_bins: int,
        max_lag: int,
    ) -> torch.Tensor:
        """
        Compute ACF of binned counts for each simulated sequence using FFT.

        Args:
            sim: SimulationResult with (time_seqs, dtime_seqs, type_seqs, mask)
            nb_bins: number of bins for discretization
            max_lag: max lag for the ACF

        Returns:
            Tensor of shape (batch_size, max_lag + 1)
        """

        hist = CorrAccumulator.create_hist(
            times=sim.time_seqs,
            mask=sim.valid_event_mask,
            nb_bins=nb_bins,
        )  # (B, nb_bins)
        # Mean-center

        hist_centered = hist - hist.mean(dim=1, keepdim=True)
        denom = torch.sum(hist_centered**2, dim=1)

        # Compute ACF using FFT
        # Pad to next power of 2 for efficiency
        n_fft = 2 ** (nb_bins - 1).bit_length() * 2

        # FFT of centered histogram
        fft_hist = torch.fft.rfft(hist_centered, n=n_fft, dim=1)

        # Power spectrum
        power_spectrum = fft_hist * torch.conj(fft_hist)

        # Inverse FFT to get autocorrelation
        acf_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=1)

        # Extract and normalize
        acf = acf_full[:, : max_lag + 1] / denom.unsqueeze(1)

        return acf

    def reset(self) -> None:
        """Reset accumulated autocorrelation data."""
        super().reset()
        self.acf_gt_mean = np.zeros(self.max_lag + 1)
        self.acf_sim_mean = np.zeros(self.max_lag + 1)
        self.batch_count = 0
