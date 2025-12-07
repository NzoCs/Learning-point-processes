# new_ltpp/models/mixins/simulation_mixin.py
"""Mixin for simulation functionality."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from tqdm import tqdm

from new_ltpp.evaluation import BatchStatisticsCollector
from new_ltpp.evaluation.accumulators.acc_types import FinalResult
from new_ltpp.shared_types import Batch, SimulationResult
from new_ltpp.utils import logger

from .base_mixin import BaseMixin

class Buffers(TypedDict):
    time: torch.Tensor
    time_delta: torch.Tensor
    event: torch.Tensor
    initial_len: int

class SimulationState(TypedDict):
    last_event_time: torch.Tensor
    current_time: float
    batch_active: torch.Tensor
    step_count: int

class SimulationMixin(BaseMixin):
    """Mixin providing simulation functionality.

    Requires: self.num_event_types, self.pad_token_id, self.initial_buffer_size, self._statistics_collector,
              self.device, self.simulation_start_time, self.simulation_end_time,
              self.simulation_batch_size, self.event_sampler, self.num_sample,
              self.compute_intensities_at_sample_times
    """

    def __init__(
        self,
        num_event_types: int,
        simulation_start_time: float,
        simulation_end_time: float,
        simulation_batch_size: int,
        initial_buffer_size: int,
        **kwargs,
    ):
        """Initialize the SimulationMixin.

        Args:
            num_sample: Number of samples for one-step prediction
            num_step_gen: Number of steps for multi-step generation
        """
        super().__init__(**kwargs)
        self.num_event_types = num_event_types
        self.simulation_start_time = simulation_start_time
        self.simulation_end_time = simulation_end_time
        self.simulation_batch_size = simulation_batch_size
        self.initial_buffer_size = initial_buffer_size
        self._statistics_collector = self.init_statistics_collector(
            self.output_dir / "distribution_comparison"
        )

    def init_statistics_collector(
        self, output_dir: Path | str
    ) -> BatchStatisticsCollector:
        """Initialize the batch statistics collector for distribution analysis.

        Args:
            output_dir: Directory where results will be saved
        """
        self._statistics_collector = BatchStatisticsCollector(
            num_event_types=self.num_event_types,
            output_dir=output_dir,
            dtime_max=self.dtime_max,
            dtime_min=0.0,
        )

        logger.info(
            f"BatchStatisticsCollector initialized with output_dir={output_dir}"
        )
        return self._statistics_collector

    def finalize_statistics(self) -> FinalResult:
        """Finalize statistics collection and generate plots/metrics.

        Returns:
            Dictionary containing statistics, metrics, and batch count
        """
        if self._statistics_collector is None:
            raise NotImplementedError(
                "No statistics collector to finalize. Initialize it first with 'init_statistics_collector'."
            )

        logger.info("Finalizing batch statistics collection...")
        results = self._statistics_collector.finalize_and_save()
        logger.info(
            f"Statistics finalized: {results.get('batch_count', 0)} batches processed"
        )
        return results



    def simulate(
        self,
        batch: Optional[Batch] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        initial_buffer_size: Optional[int] = None,
    ) -> SimulationResult:
        """Simulate event sequences using the model.

        Args:
            batch: Optional initial batch to condition on
            start_time: Simulation start time
            end_time: Simulation end time
            batch_size: Number of sequences to simulate
            initial_buffer_size: Initial size of simulation buffers (will grow dynamically)

        Returns:
            Tuple of (time_seq, time_delta_seq, event_seq, mask)
        """
        # Set default values
        if start_time is None:
            start_time = self.simulation_start_time
            if start_time is None:
                raise ValueError(
                    "start_time must be provided or set via set_simulation_times()"
                )

        if end_time is None:
            end_time = self.simulation_end_time
            if end_time is None:
                raise ValueError(
                    "end_time must be provided or set via set_simulation_times()"
                )

        if batch_size is None:
            batch_size = self.simulation_batch_size

        if initial_buffer_size is None:
            initial_buffer_size = self.initial_buffer_size

        # Initialize sequences
        if batch is None:
            batch = self._create_empty_batch(batch_size)
        else:
            batch_size = batch.time_seqs.size(0)

        start_times, end_times = self.compute_start_end_time(batch.time_seqs, batch.seq_non_pad_mask)

        # Pre-allocate buffers
        buffers = self._allocate_simulation_buffers(batch, initial_buffer_size)

        # Initialize tracking state
        sim_state = self._initialize_simulation_state(batch, batch_size)


        # Run simulation loop
        self._run_simulation_loop(buffers, sim_state, start_times, end_times)
        # Extract and return results
        return self._extract_simulation_results(
            buffers, sim_state, start_times, end_times
        )

    def compute_start_end_time(self, time_seqs: torch.Tensor, seq_non_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute start and end times for simulation.

        Returns:
            Tuple of (start_time, end_time)
        """

        # put the padded items to infinity so that they won't affect min computation
        time_seqs[~seq_non_pad_mask] = float("inf")
        start_times = time_seqs.min(dim=1).values

        # put it back to original values
        time_seqs[~seq_non_pad_mask] = 0.0
        end_times = time_seqs.max(dim=1).values

        return start_times, end_times

    def simulate_one_step(
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_non_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate one step of event generation.

        Args:
            time_seqs: Time sequence [batch_size, seq_len]
            time_delta_seqs: Time delta sequence [batch_size, seq_len]
            type_seqs: Event type sequence [batch_size, seq_len]
            mode: "train" or "simulation" mode

        Returns:
            Tuple of (predicted_dtimes, predicted_types)
        """

        # Draw next time
        accepted_dtimes, weights = self.get_event_sampler().draw_next_time_one_step(
            time_seqs,
            time_delta_seqs,
            type_seqs,
            seq_non_pad_mask,
            self.compute_intensities_at_sample_dtimes,
            num_sample=1,
            compute_last_step_only=True,
        )

        # Estimate next time delta
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        batch_size, num_mark = time_seqs.size(0), self.num_event_types

        # Select next event type based on intensities
        intensities_at_times = self.compute_intensities_at_sample_dtimes(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
            seq_non_pad_mask=seq_non_pad_mask,
            sample_dtimes=dtimes_pred[:, :, None],
            compute_last_step_only=True,
        ).view(batch_size, num_mark)

        total_intensities = intensities_at_times.sum(dim=-1)

        if torch.any(total_intensities == 0):
            raise ValueError("Total intensities is null, simulation stops.")

        probs = intensities_at_times / total_intensities[:, None]
        type_pred = torch.multinomial(probs, num_samples=1)

        return dtimes_pred, type_pred

    def _create_empty_batch(self, batch_size: int) -> Batch:
        """Create an empty batch for simulation initialization."""
        return Batch(
            time_seqs=torch.zeros(
                batch_size, 2, device=self.device, dtype=torch.float32
            ),
            time_delta_seqs=torch.zeros(
                batch_size, 2, device=self.device, dtype=torch.float32
            ),
            type_seqs=torch.zeros(batch_size, 2, device=self.device, dtype=torch.long),
            seq_non_pad_mask=torch.ones(
                batch_size, 2, device=self.device, dtype=torch.bool
            ),
        )

    def _allocate_simulation_buffers(self, batch: Batch, initial_buffer_size: int) -> Buffers:
        """Allocate buffers for simulation."""
        batch_size = batch.time_seqs.size(0)
        initial_len = batch.time_seqs.size(1)
        max_seq_len = initial_buffer_size + initial_len

        time_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        time_delta_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        event_buffer = torch.zeros(
            batch_size, max_seq_len, device=self.device, dtype=torch.long
        ).contiguous()

        # Copy initial sequences
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
        """Initialize simulation tracking state."""
        # Track last event time for each mark
        last_event_time = torch.zeros(
            (batch_size, self.num_event_types), device=self.device, dtype=torch.float32
        )

        time_seq = batch.time_seqs
        event_seq = batch.type_seqs

        # Compute last event time for each mark using vectorized operations
        for mark in range(self.num_event_types):
            mark_mask = event_seq == mark
            if mark_mask.any():
                masked_times = time_seq.masked_fill(~mark_mask, float("-inf"))
                max_times, _ = masked_times.max(dim=1)
                valid_mask = max_times != float("-inf")
                last_event_time[valid_mask, mark] = max_times[valid_mask]

        return SimulationState(
            last_event_time=last_event_time,
            current_time=batch.time_seqs[:, -1].min().item(),
            batch_active=torch.ones(
                batch_size, dtype=torch.bool, device=self.device
            ),
            step_count=0,
        )

    def _reallocate_buffers(
        self, buffers: Buffers, current_max_len: int
    ) -> int:
        """Reallocate buffers with double the size.

        Args:
            buffers: Buffers TypedDict containing current buffers
            current_max_len: Current maximum sequence length

        Returns:
            New maximum sequence length
        """
        batch_size = buffers["time"].size(0)
        new_max_seq_len = current_max_len * 2

        # Allocate new larger buffers
        new_time_buffer = torch.zeros(
            batch_size, new_max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        new_time_delta_buffer = torch.zeros(
            batch_size, new_max_seq_len, device=self.device, dtype=torch.float32
        ).contiguous()
        new_event_buffer = torch.zeros(
            batch_size, new_max_seq_len, device=self.device, dtype=torch.long
        ).contiguous()

        # Copy existing data
        new_time_buffer[:, :current_max_len].copy_(buffers["time"])
        new_time_delta_buffer[:, :current_max_len].copy_(buffers["time_delta"])
        new_event_buffer[:, :current_max_len].copy_(buffers["event"])

        # Update buffers
        buffers["time"] = new_time_buffer
        buffers["time_delta"] = new_time_delta_buffer
        buffers["event"] = new_event_buffer

        logger.info(f"Reallocated buffers to size {new_max_seq_len}")

        return new_max_seq_len
    
    def _run_simulation_loop(
        self,
        buffers: Buffers,
        sim_state: SimulationState,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
    ) -> None:
        
        """Run the main simulation loop. Simulates starting from the end of initial sequences, so at end_time to start_time + end_time for each sequence.
        
        Args:   
            buffers: Buffers TypedDict containing simulation buffers
            sim_state: SimulationState TypedDict tracking simulation state
            start_times: Tensor of start times for each sequence
            end_times: Tensor of end times for each sequence"""

        initial_len = buffers["initial_len"]
        max_seq_len = buffers["time"].size(1)

        # 1. Calcul du temps de fin absolu pour chaque séquence
        # We start at end_time and simulate for the same duration (end_time - start_time) for each sequence
        absolute_end_times = end_times + (end_times - start_times)
        max_absolute_end_time = absolute_end_times.max().item()

        # 2. Application du décalage (start_time) et initialisation de current_time
        if sim_state["step_count"] == 0:
            if initial_len > 0:
                # Ajout du décalage start_time aux temps initiaux (si initialement relatifs à t=0)
                # Cette ligne suppose que le code appelant n'a pas encore appliqué le décalage.
                buffers["time"][:, :initial_len] += start_times.unsqueeze(1)
                sim_state["last_event_time"] += start_times.unsqueeze(1)
                sim_state["current_time"] = buffers["time"][:, initial_len - 1].min().item()
            else:
                # Si le buffer est vide (initial_len=0), le temps courant minimum est le temps de début minimum
                sim_state["current_time"] = start_times.min().item()
        with torch.no_grad():
            pbar = tqdm(total=max_absolute_end_time, desc="Simulation", leave=False)
            pbar.n = min(sim_state["current_time"], max_absolute_end_time)
            pbar.refresh()


        while sim_state["batch_active"].any():
            
            # Get active sequences
            active_indices = sim_state["batch_active"].nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                break

            current_len = initial_len + sim_state["step_count"]

            # Check if we need to reallocate buffers (double the size)
            if current_len >= max_seq_len - 1:
                max_seq_len = self._reallocate_buffers(buffers, max_seq_len)

            # Predict on active sequences
            active_time_seq = buffers["time"][active_indices, :current_len]
            active_time_delta = buffers["time_delta"][active_indices, :current_len]
            active_event_seq = buffers["event"][active_indices, :current_len]
            active_seq_non_pad_mask = (active_event_seq != self.pad_token_id)

            dtimes_pred, type_pred = self.simulate_one_step(
                active_time_seq, active_time_delta, active_event_seq, active_seq_non_pad_mask
            )

            # Calculate new times (Active_time_seq[-1] est maintenant un temps absolu)
            new_times = active_time_seq[:, -1:] + dtimes_pred

            # Update last_event_time
            type_pred_flat = type_pred.squeeze(-1)
            last_times_flat = active_time_seq[:, -1]
            sim_state["last_event_time"][
                active_indices, type_pred_flat
            ] = last_times_flat


            # Update buffers
            buffers["time"][active_indices, current_len] = new_times.squeeze(-1)
            buffers["time_delta"][active_indices, current_len] = dtimes_pred.squeeze(-1)
            buffers["event"][active_indices, current_len] = type_pred.squeeze(-1)

            # Update current time (minimum across all active sequences)
            sim_state["current_time"] = new_times.min().item()
            
            # Désactivation des séquences ayant atteint leur temps de fin individuel
            active_end_times = absolute_end_times[active_indices].unsqueeze(-1)
            exceed_time_mask = new_times >= active_end_times
            
            if exceed_time_mask.any():
                exceed_indices = active_indices[exceed_time_mask.squeeze(-1)]
                sim_state["batch_active"][exceed_indices] = False

            sim_state["step_count"] += 1

            if sim_state["step_count"] % 50 == 0:
                pbar.n = min(sim_state["current_time"], max_absolute_end_time)
                pbar.refresh()

        pbar.close()

    def _extract_simulation_results(
        self, buffers: Buffers, sim_state: SimulationState, start_times: torch.Tensor, end_times: torch.Tensor
    ) -> SimulationResult:
        """Extract final simulation results from buffers."""
        initial_len = buffers["initial_len"]
        current_len = initial_len + sim_state["step_count"]

        final_time_seq = buffers["time"][:, initial_len:current_len]
        final_time_delta = buffers["time_delta"][:, initial_len:current_len]
        final_event_seq = buffers["event"][:, initial_len:current_len]

        absolute_end_times = end_times + (end_times - start_times) 

        # Create final mask
        simul_mask = torch.logical_and(
            final_time_seq >= end_times.unsqueeze(-1), final_time_seq <= absolute_end_times.unsqueeze(-1)
        )

        return SimulationResult(
            final_time_seq, final_time_delta, final_event_seq, simul_mask
        )