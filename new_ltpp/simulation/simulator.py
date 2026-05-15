"""Simulator — standalone engine for TPP event sequence generation.

Extracted from SimulationMixin. Takes a model as dependency via ISimulableModel
and owns the simulation loop, buffer management, and statistics collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypedDict

import torch
from tqdm import tqdm

from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger
from new_ltpp.evaluation.accumulators.acc_types import FinalResult

if TYPE_CHECKING:
    from new_ltpp.evaluation import BatchStatisticsCollector
    from new_ltpp.models.model_protocol import ISimulableModel
    from new_ltpp.configs.runner_config import StatisticalTestConfig


class Buffers(TypedDict):
    time: torch.Tensor
    time_delta: torch.Tensor
    event: torch.Tensor
    initial_len: int


class SimulationState(TypedDict):
    current_time: torch.Tensor
    batch_active: torch.Tensor
    step_count: int


class Simulator:
    """Standalone simulation engine for TPP models.

    Owns the full simulation loop, buffer management, and statistics collection.
    Requires a model implementing ISimulableModel (compute_intensities, get_event_sampler).

    Args:
        model: A model implementing ISimulableModel.
        start_time: Default simulation start time.
        end_time: Default simulation end time.
        batch_size: Default number of sequences to simulate.
        initial_buffer_size: Initial pre-allocated buffer length per sequence.
        statistical_test_config: Optional config dict forwarded to BatchStatisticsCollector.
    """

    def __init__(
        self,
        model: "ISimulableModel",
        statistical_test_config: Optional["StatisticalTestConfig"] = None,
    ) -> None:
        self._model = model
        self.statistical_test_config = statistical_test_config
        self._statistics_collector: Optional["BatchStatisticsCollector"] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        batch: Batch,
    ) -> SimulationResult:
        """Simulate event sequences using the model.

        Args:
            batch: Optional initial batch to condition on.

        Returns:
            SimulationResult (Batch alias) with generated sequences.
        """

        batch_size = batch.time_seqs.size(0)

        start_times, end_times = self._compute_start_end_time(
            batch.time_seqs, batch.valid_event_mask
        )

        initial_buffer_size = batch.time_seqs.size(1)

        buffers = self._allocate_simulation_buffers(batch, initial_buffer_size)
        sim_state = self._initialize_simulation_state(batch, batch_size)
        self._run_simulation_loop(buffers, sim_state, start_times, end_times)
        return self._extract_simulation_results(
            buffers, sim_state, start_times, end_times
        )

    def simulate_from_scratch(
        self,
        num_sequences: int,
    ) -> SimulationResult:
        """Simulate event sequences from scratch (no conditioning).

        Args:
            num_sequences: Number of sequences to simulate (defaults to self.batch_size).

        Returns:
            SimulationResult (Batch alias) with generated sequences.
        """
        empty_batch = self._create_empty_batch(num_sequences)
        return self.simulate(empty_batch)

    def init_statistics_collector(self, base_dir: Path | str) -> None:
        """Initialize the BatchStatisticsCollector.

        Args:
            base_dir: Base directory where statistics and plots will be saved.
        """
        from new_ltpp.evaluation import BatchStatisticsCollector

        model = self._model
        stat_cfg = self.statistical_test_config

        if stat_cfg is None:
            raise ValueError(
                "Statistical test config is required to initialize statistics collector."
            )

        metadata: Dict[str, Any] = {
            "model": {
                "class_name": model.__class__.__name__,
                "num_event_types": model.num_event_types,
                "pad_token_id": model.pad_token_id,
            },
            "statistical_test_config": stat_cfg.model_dump(),
            "device": str(model.device),
        }

        self._statistics_collector = BatchStatisticsCollector(
            num_event_types=model.num_event_types,
            base_dir=base_dir,
            dtime_max=model.dtime_max,
            dtime_min=0.0,
            statistical_test_config=stat_cfg,
            metadata=metadata,
        )
        logger.info(f"Simulator: BatchStatisticsCollector initialized at {base_dir}")

    def finalize_statistics(self) -> FinalResult:
        """Finalize statistics collection and generate plots/metrics.

        Returns:
            FinalResult with statistics, metrics, and batch_count.
        """
        if self._statistics_collector is None:
            raise RuntimeError(
                "No statistics collector. Call init_statistics_collector() first."
            )
        logger.info("Simulator: finalizing batch statistics...")
        results = self._statistics_collector.finalize_and_save(generate_plots=True)
        logger.info(
            f"Simulator: statistics finalized ({results.get('batch_count', 0)} batches)"
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers (previously private methods of SimulationMixin)
    # ------------------------------------------------------------------

    def _compute_start_end_time(
        self, time_seqs: torch.Tensor, valid_event_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._model.device
        time_seqs = time_seqs.to(device)
        valid_event_mask = valid_event_mask.to(device)
        time_seqs[~valid_event_mask] = float("inf")
        start_times = time_seqs.min(dim=1).values
        time_seqs[~valid_event_mask] = 0.0
        end_times = time_seqs.max(dim=1).values
        return start_times, end_times

    def _create_empty_batch(self, batch_size: int) -> Batch:
        device = self._model.device
        return Batch(
            time_seqs=torch.zeros(batch_size, 2, device=device, dtype=torch.float32),
            time_delta_seqs=torch.zeros(
                batch_size, 2, device=device, dtype=torch.float32
            ),
            type_seqs=torch.zeros(batch_size, 2, device=device, dtype=torch.long),
            valid_event_mask=torch.ones(batch_size, 2, device=device, dtype=torch.bool),
        )

    def _allocate_simulation_buffers(
        self, batch: Batch, initial_buffer_size: int
    ) -> Buffers:
        device = self._model.device
        batch_size = batch.time_seqs.size(0)
        initial_len = batch.time_seqs.size(1)
        max_seq_len = initial_buffer_size + initial_len

        time_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        time_delta_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        event_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.long
        ).contiguous()

        time_buffer[:, :initial_len].copy_(batch.time_seqs)
        time_delta_buffer[:, :initial_len].copy_(batch.time_delta_seqs)
        event_buffer[:, :initial_len].copy_(batch.type_seqs)

        return Buffers(
            time=time_buffer,
            time_delta=time_delta_buffer,
            event=event_buffer,
            initial_len=initial_len,
        )

    def _initialize_simulation_state(
        self, batch: Batch, batch_size: int
    ) -> SimulationState:
        device = self._model.device

        return SimulationState(
            current_time=batch.time_seqs[:, -1].min(),
            batch_active=torch.ones(batch_size, dtype=torch.bool, device=device),
            step_count=0,
        )

    def _simulate_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self._model
        accepted_dtimes, weights = model.get_event_sampler().draw_next_time_one_step(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            valid_event_mask,
            model.compute_intensities_at_sample_dtimes,
            num_sample=1,
            compute_last_step_only=True,
        )

        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        batch_size, num_mark = time_seqs.size(0), model.num_event_types

        intensities_at_times = model.compute_intensities_at_sample_dtimes(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
            sample_dtimes=dtimes_pred[:, :, None],
            compute_last_step_only=True,
        ).view(batch_size, num_mark)

        total_intensities = intensities_at_times.sum(dim=-1)
        if torch.any(total_intensities == 0):
            raise ValueError("Total intensities is null, simulation stops.")

        probs = intensities_at_times / total_intensities[:, None]
        type_pred = torch.multinomial(probs, num_samples=1)
        return dtimes_pred, type_pred

    def _reallocate_buffers(self, buffers: Buffers, current_max_len: int) -> int:
        device = self._model.device
        batch_size = buffers["time"].size(0)
        new_max_seq_len = current_max_len * 2

        new_time = torch.zeros(
            batch_size, new_max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        new_delta = torch.zeros(
            batch_size, new_max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        new_event = torch.zeros(
            batch_size, new_max_seq_len, device=device, dtype=torch.long
        ).contiguous()

        new_time[:, :current_max_len].copy_(buffers["time"])
        new_delta[:, :current_max_len].copy_(buffers["time_delta"])
        new_event[:, :current_max_len].copy_(buffers["event"])

        buffers["time"] = new_time
        buffers["time_delta"] = new_delta
        buffers["event"] = new_event

        logger.info(f"Simulator: reallocated buffers to size {new_max_seq_len}")
        return new_max_seq_len

    def _run_simulation_loop(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
    ) -> None:
        initial_len = buffers["initial_len"]
        max_seq_len = buffers["time"].size(1)
        pad_token_id = self._model.pad_token_id

        sim_window = end_times - start_times
        sim_end_times = end_times + sim_window
        max_sim_end_time = sim_end_times.max()

        if sim_state["step_count"] == 0:
            if initial_len > 0:
                buffers["time"][:, :initial_len] += start_times.unsqueeze(1)
                sim_state["current_time"] = end_times.min()
            else:
                sim_state["current_time"] = start_times.min()

        with (
            torch.no_grad(),
            tqdm(total=max_sim_end_time.item(), desc="Simulation", leave=False) as pbar,
        ):
            pbar.n = min(sim_state["current_time"].item(), max_sim_end_time.item())
            pbar.refresh()

            while sim_state["batch_active"].any():
                active_indices = sim_state["batch_active"].nonzero(as_tuple=True)[0]
                if len(active_indices) == 0:
                    break

                current_len = initial_len + sim_state["step_count"]

                if current_len >= max_seq_len - 1:
                    max_seq_len = self._reallocate_buffers(buffers, max_seq_len)

                active_time_seq = buffers["time"][active_indices, :current_len]
                active_time_delta = buffers["time_delta"][active_indices, :current_len]
                active_event_seq = buffers["event"][active_indices, :current_len]
                active_valid_event_mask = active_event_seq != pad_token_id

                dtimes_pred, type_pred = self._simulate_one_step(
                    active_time_seq,
                    active_time_delta,
                    active_event_seq,
                    active_valid_event_mask,
                )

                new_times = active_time_seq[:, -1:] + dtimes_pred

                buffers["time"][active_indices, current_len] = new_times.squeeze(-1)
                buffers["time_delta"][active_indices, current_len] = (
                    dtimes_pred.squeeze(-1)
                )
                buffers["event"][active_indices, current_len] = type_pred.squeeze(-1)

                sim_state["current_time"] = new_times.min()

                active_end_times = sim_end_times[active_indices].unsqueeze(-1)
                exceed_time_mask = new_times >= active_end_times
                if exceed_time_mask.any():
                    exceed_indices = active_indices[exceed_time_mask.squeeze(-1)]
                    sim_state["batch_active"][exceed_indices] = False

                sim_state["step_count"] += 1

                if sim_state["step_count"] % 50 == 0:
                    pbar.n = min(
                        sim_state["current_time"].item(), sim_end_times.max().item()
                    )
                    pbar.refresh()

    def _extract_simulation_results(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
    ) -> SimulationResult:
        pad_token_id = self._model.pad_token_id
        initial_len = buffers["initial_len"]
        current_len = initial_len + sim_state["step_count"]

        final_time_seq = buffers["time"][:, initial_len:current_len]
        final_time_delta = buffers["time_delta"][:, initial_len:current_len]
        final_event_seq = buffers["event"][:, initial_len:current_len]

        # compute sim window and end times
        sim_window = end_times - start_times
        sim_end_times = end_times + sim_window
        sim_start_times = end_times

        simul_mask = torch.logical_and(
            final_time_seq >= sim_start_times.unsqueeze(-1),
            final_time_seq <= sim_end_times.unsqueeze(-1),
        )

        final_event_seq[~simul_mask] = pad_token_id

        return SimulationResult(
            time_seqs=final_time_seq,
            time_delta_seqs=final_time_delta,
            type_seqs=final_event_seq,
            valid_event_mask=simul_mask,
        )
