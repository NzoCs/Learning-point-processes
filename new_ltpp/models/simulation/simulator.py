"""Simulator — standalone engine for TPP event sequence generation.

Extracted from SimulationMixin. Takes a model as dependency via ISimulableModel
and owns the simulation loop, buffer management, and statistics collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from new_ltpp.evaluation.accumulators.acc_types import FinalResult
from new_ltpp.shared_types import (
    STOPPED_OBSERVATION_PROTOCOL,
    Batch,
    SimulationResult,
    uses_stopped_event_budget,
)
from new_ltpp.utils import logger

if TYPE_CHECKING:
    from new_ltpp.configs.runner_config import StatisticalTestConfig
    from new_ltpp.evaluation import BatchStatisticsCollector
    from new_ltpp.models.model_protocol import ISimulableModel


class Buffers(TypedDict):
    time: torch.Tensor
    time_delta: torch.Tensor
    event: torch.Tensor
    initial_len: int


class SimulationState(TypedDict):
    step_count: int
    no_next_event_mask: torch.Tensor


class PredictionSession:
    """Prediction statistics and artefacts, independent of the sampling algorithm."""

    def __init__(
        self,
        model: "ISimulableModel",
        statistical_test_config: Optional["StatisticalTestConfig"] = None,
        conditional_evaluation=None,
    ) -> None:
        self.last_result: Optional[SimulationResult] = None
        self._model = model
        self.statistical_test_config = statistical_test_config
        self.conditional_evaluation = conditional_evaluation
        self._statistics_collector: Optional["BatchStatisticsCollector"] = None

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
        if self.conditional_evaluation is not None:
            metadata["conditional_evaluation"] = (
                self.conditional_evaluation.model_dump()
            )
            metadata["statistical_protocol"] = (
                "within_history_pair_swaps_joint_kernel_v1"
            )
        complete = uses_stopped_event_budget(model.__class__.__name__)
        if complete:
            metadata["observation_protocol"] = STOPPED_OBSERVATION_PROTOCOL

        self._statistics_collector = BatchStatisticsCollector(
            num_event_types=model.num_event_types,
            base_dir=base_dir,
            dtime_max=model.dtime_max,
            dtime_min=0.0,
            statistical_test_config=stat_cfg,
            metadata=metadata,
            conditional=self.conditional_evaluation is not None,
            complete_stopped=complete,
            stopped_event_budget=(
                self.conditional_evaluation.future_events
                if self.conditional_evaluation is not None
                else None
            ),
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


class Simulator(PredictionSession):
    """Neural simulation engine; parametric requests delegate to their native generator."""

    def __init__(
        self, model, statistical_test_config=None, conditional_evaluation=None
    ):
        super().__init__(model, statistical_test_config, conditional_evaluation)
        self._intensity_fn = model.compute_intensities_at_sample_dtimes

    def simulate(
        self,
        batch: Batch,
        num_events_to_simulate: Optional[int] = None,
    ) -> SimulationResult:
        """Simulate event sequences using the model.

        Args:
            batch: Initial batch to condition on. Can be empty (length 0) for unconditional generation.
            num_events_to_simulate: Number of events to generate. If None, defaults to the sequence length in the batch.

        Returns:
            SimulationResult (Batch alias) with generated sequences.
        """
        if getattr(self._model, "model_family", None) == "parametric":
            self.last_result = self._model.simulate(batch, num_events_to_simulate)
            return self.last_result
        if num_events_to_simulate is None:
            # We simulate exactly as many events as the sequence length in the batch by default.
            num_events_to_simulate = batch.time_seqs.size(1)

        buffers = self._allocate_simulation_buffers(batch, num_events_to_simulate)
        sim_state = self._initialize_simulation_state()
        self._run_simulation_loop(buffers, sim_state, num_events_to_simulate)
        self.last_result = self._extract_simulation_results(buffers, sim_state)
        return self.last_result

    # ------------------------------------------------------------------
    # Internal helpers (previously private methods of SimulationMixin)
    # ------------------------------------------------------------------

    def _allocate_simulation_buffers(
        self, batch: Batch, num_events_to_simulate: int
    ) -> Buffers:
        device = self._model.device
        batch_size = batch.time_seqs.size(0)
        initial_len = batch.time_seqs.size(1)
        max_seq_len = initial_len + num_events_to_simulate

        time_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        time_delta_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.float32
        ).contiguous()
        event_buffer = torch.zeros(
            batch_size, max_seq_len, device=device, dtype=torch.long
        ).contiguous()
        event_buffer.fill_(self._model.pad_token_id)

        time_buffer[:, :initial_len].copy_(batch.time_seqs)
        time_delta_buffer[:, :initial_len].copy_(batch.time_delta_seqs)
        event_buffer[:, :initial_len].copy_(batch.type_seqs)

        return Buffers(
            time=time_buffer,
            time_delta=time_delta_buffer,
            event=event_buffer,
            initial_len=initial_len,
        )

    def _initialize_simulation_state(self) -> SimulationState:
        return SimulationState(
            step_count=0, no_next_event_mask=torch.empty(0, dtype=torch.bool)
        )

    def _simulate_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        valid_event_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self._model
        direct_sampler = getattr(model, "sample_next_event", None)
        if callable(direct_sampler):
            return direct_sampler(
                time_seqs=time_seqs,
                time_delta_seqs=time_delta_seqs,
                type_seqs=type_seqs,
                valid_event_mask=valid_event_mask,
            )
        accepted_dtimes, weights = model.get_event_sampler().draw_next_time_one_step(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            valid_event_mask,
            self._intensity_fn,
            num_sample=1,
            compute_last_step_only=True,
        )

        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        batch_size, num_mark = time_seqs.size(0), model.num_event_types

        intensities_at_times = self._intensity_fn(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            valid_event_mask=valid_event_mask,
            sample_dtimes=torch.where(torch.isfinite(dtimes_pred), dtimes_pred, 0)[
                :, :, None
            ],
            compute_last_step_only=True,
        ).view(batch_size, num_mark)

        total_intensities = intensities_at_times.sum(dim=-1)
        finite = torch.isfinite(dtimes_pred).squeeze(-1)
        if torch.any((total_intensities == 0) & finite):
            raise ValueError("Total intensities is null, simulation stops.")

        probs = torch.where(
            finite[:, None], intensities_at_times, torch.ones_like(intensities_at_times)
        )
        probs = probs / probs.sum(-1, keepdim=True)
        type_pred = torch.multinomial(probs, num_samples=1)
        type_pred = torch.where(finite[:, None], type_pred, model.pad_token_id)
        return dtimes_pred, type_pred

    def _run_simulation_loop(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
        num_events_to_simulate: int,
    ) -> None:
        initial_len = buffers["initial_len"]
        pad_token_id = self._model.pad_token_id
        alive = torch.ones(
            buffers["time"].shape[0], dtype=torch.bool, device=buffers["time"].device
        )
        sim_state["no_next_event_mask"] = ~alive

        with (
            torch.no_grad(),
            tqdm(
                total=num_events_to_simulate,
                desc="Simulation",
                leave=False,
                disable=True,
            ) as pbar,
        ):
            for step in range(num_events_to_simulate):
                if not alive.any():
                    break
                rows = alive.nonzero().squeeze(-1)
                current_len = initial_len + sim_state["step_count"]

                active_time_seq = buffers["time"][rows, :current_len]
                active_time_delta = buffers["time_delta"][rows, :current_len]
                active_event_seq = buffers["event"][rows, :current_len]
                active_valid_event_mask = active_event_seq != pad_token_id

                dtimes_pred, type_pred = self._simulate_one_step(
                    active_time_seq,
                    active_time_delta,
                    active_event_seq,
                    active_valid_event_mask,
                )
                finite = torch.isfinite(dtimes_pred).squeeze(-1)
                if torch.isnan(dtimes_pred).any() or (dtimes_pred < 0).any():
                    raise ValueError("Invalid sampled waiting time")
                alive[rows[~finite]] = False
                sim_state["no_next_event_mask"] = ~alive

                if current_len > 0:
                    new_times = active_time_seq[:, -1:] + dtimes_pred
                else:
                    new_times = dtimes_pred

                buffers["time"][rows[finite], current_len] = new_times.squeeze(-1)[
                    finite
                ]
                buffers["time_delta"][rows[finite], current_len] = dtimes_pred.squeeze(
                    -1
                )[finite]
                buffers["event"][rows[finite], current_len] = type_pred.squeeze(-1)[
                    finite
                ]

                sim_state["step_count"] += 1
                pbar.update(1)

    def _extract_simulation_results(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
    ) -> SimulationResult:
        initial_len = buffers["initial_len"]
        current_len = initial_len + sim_state["step_count"]

        final_time_seq = buffers["time"][:, initial_len:current_len].clone()
        final_time_delta = buffers["time_delta"][:, initial_len:current_len].clone()
        final_event_seq = buffers["event"][:, initial_len:current_len].clone()

        # Shift times so that the simulation conceptually starts at t=0
        if initial_len > 0:
            offset = buffers["time"][:, initial_len - 1].unsqueeze(-1)
            final_time_seq = final_time_seq - offset

        # Valid mask is True for all generated events
        simul_mask = final_event_seq != self._model.pad_token_id
        final_time_seq = torch.where(simul_mask, final_time_seq, 0)

        return SimulationResult(
            time_seqs=final_time_seq,
            time_delta_seqs=final_time_delta,
            type_seqs=final_event_seq,
            valid_event_mask=simul_mask,
            no_next_event_mask=sim_state["no_next_event_mask"],
        )
