# new_ltpp/models/mixins/visualization_mixin.py
"""Mixin for visualization and analysis methods."""

import os
from collections import defaultdict
from typing import Dict, Tuple

import torch
from matplotlib import pyplot as plt

from new_ltpp.utils import logger, save_json

from .simulation_mixin import SimulationMixin


class VisualizationMixin(SimulationMixin):
    """Mixin providing visualization and analysis functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def intensity_graph(
        self,
        save_dir: str,
        *,
        precision: int = 100,
        plot: bool = False,
        save_plot: bool = True,
        save_data: bool = True,
        **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """Generate and visualize intensity curves for the model.

        Args:
            precision: Number of interpolation points between events
            plot: Whether to display the plots
            save_plot: Whether to save plots to disk
            save_data: Whether to save intensity data to disk
            save_dir: Directory for saving outputs

        Returns:
            Tuple of (intensities, time_points, marked_times)
        """
        start_time = self.simulation_start_time
        end_time = self.simulation_end_time

        # Get or generate simulation data
        time_seq, time_delta_seq, type_seq = self._get_simulation_data(
            start_time, end_time
        )

        # Generate time points for intensity calculation
        time_points, time_deltas_sample = self._generate_intensity_time_points(
            time_seq, time_delta_seq, precision
        )

        # Calculate intensities
        intensities = self._calculate_intensities(
            time_seq, time_delta_seq, type_seq, time_deltas_sample
        )

        intensities_at_times = self._calculate_intensities(
            time_seq, time_delta_seq, type_seq, time_delta_seq[:, 1:, None]
        )

        # Flatten for analysis
        time_flat = time_points.view(-1)
        intensities_flat = intensities[0, ...].view(-1, self.num_event_types)

        # Collect marked event times
        marked_times, intensities_at_marked_times = self._collect_marked_times_intensities(
            intensities_at_times, time_seq[:, 1:], type_seq[:, 1:], self.num_event_types
            )

        # Save data if requested
        if save_data:
            self._save_intensity_data(
                time_flat,
                intensities_flat,
                marked_times,
                save_dir,
                precision,
                start_time,
                end_time,
                self.num_event_types,
            )

        # Plot if requested
        if plot or save_plot:
            self._plot_intensity_graphs(
                time_flat,
                marked_times,                
                intensities_flat,                
                intensities_at_marked_times,
                self.num_event_types,
                save_dir,
                plot,
                save_plot,
            )

        return intensities_flat, time_flat, marked_times

    def get_model_metadata(self) -> Dict:
        """Get metadata about the model for simulation purposes.

        Returns:
            Dictionary containing model metadata
        """
        metadata = {
            "model_type": self.__class__.__name__,
            "num_event_types": self.num_event_types,
        }

        # Add hyperparameters if available
        if hasattr(self, "hparams"):
            for key, value in self.hparams.items():
                if (
                    isinstance(value, (int, float, str, bool, list, dict))
                    or value is None
                ):
                    if key not in metadata:
                        metadata[key] = value

        return metadata

    def _get_simulation_data(self, start_time: float, end_time: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get or generate simulation data for visualization.
        
        Args:
            start_time: Start time for simulation
            end_time: End time for simulation
        Returns:
            Tuple of (time_seq, time_delta_seq, type_seq) tensors [seq_len]    
        """

        simul_result = (
            self.simulate(start_time=start_time, end_time=end_time, batch_size=1)
        )
        time_seq = simul_result.time_seqs
        time_delta_seq = simul_result.dtime_seqs
        type_seq = simul_result.type_seqs

        return time_seq, time_delta_seq, type_seq

    def _generate_intensity_time_points(self, time_seq: torch.Tensor, time_delta_seq: torch.Tensor, precision: int):
        """Generate time points for intensity calculation.
        
        Args:
            time_seq: [1, seq_len]
            time_delta_seq: [1, seq_len]
            precision: Number of interpolation points between events

        Returns:
            Tuple of (time_points, time_deltas_sample) tensors [1, seq_len - 1, precision]
        """

        ratios = (
            torch.linspace(start=0.0, end=1.0, steps=precision, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        time_starts = time_seq[:, :-1]
        time_deltas = time_delta_seq[:, 1:]
        time_deltas_sample = time_deltas.unsqueeze(-1) * ratios

        time_diffs = time_seq.diff()
        time_points = time_starts.unsqueeze(-1) + time_diffs.unsqueeze(-1) * ratios

        return time_points, time_deltas_sample

    def _calculate_intensities(
        self, time_seqs: torch.Tensor, time_delta_seqs: torch.Tensor, type_seqs: torch.Tensor, time_deltas_sample: torch.Tensor
    ):
        """Calculate intensities on time grid.
        
        Args:
            time_seqs: [1, seq_len]
            time_delta_seqs: [1, seq_len]
            type_seqs: [1, seq_len]
            time_deltas_sample: [1, seq_len - 1, precision]
            
        Returns:
            intensities: [1, seq_len - 1, precision, num_event_types]
        """

        with torch.no_grad():

            # Compute intensities at sampled times, excluding initial zero time because we could not compute dt there
            intensities = self.compute_intensities_at_sample_times(
                time_seqs=time_seqs[:, 1:],
                time_delta_seqs=time_delta_seqs[:, 1:],
                type_seqs=type_seqs[:, 1:],
                sample_dtimes=time_deltas_sample,
            )
        return intensities.detach().clone()

    def _collect_marked_times_intensities(
            self, intensities_at_times: torch.Tensor, time_seqs: torch.Tensor, type_seqs: torch.Tensor, num_mark: int
            ) -> Tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Collect times where each event type occurs."""

        marked_times = {}
        intensities_at_marked_times = {}
        time_seqs_flat = time_seqs.squeeze(0)
        type_seqs_flat = type_seqs.squeeze(0)

        valid_mask = (time_seqs_flat != 0)

        for i in range(num_mark):
            mask = (type_seqs_flat == i) & valid_mask
            
            if mask.any():
                marked_times[i] = time_seqs_flat[mask]
                intensities_at_marked_times[i] = intensities_at_times[0, :, 0, i][mask]
            else:
                marked_times[i] = torch.empty(0, device=self.device)
                intensities_at_marked_times[i] = torch.empty(0, device=self.device)

        return marked_times, intensities_at_marked_times

    def _save_intensity_data(
        self,
        time_flat,
        intensities_flat,
        marked_times,
        save_dir,
        precision,
        start_time,
        end_time,
        num_mark,
    ):
        """Save intensity data to JSON file."""
        os.makedirs(save_dir, exist_ok=True)

        intensity_data = {
            "time_points": time_flat.cpu().detach().numpy().tolist(),
            "intensities": intensities_flat.cpu().detach().numpy().tolist(),
            "marked_times": {
                str(dim): times.cpu().detach().numpy().tolist()
                for dim, times in marked_times.items()
            },
            "metadata": {
                "precision": precision,
                "start_time": start_time,
                "end_time": end_time,
                "num_event_types": num_mark,
                "model_type": self.__class__.__name__,
            },
        }

        data_file = os.path.join(
            save_dir, f"{self.__class__.__name__}_intensity_data.json"
        )
        save_json(intensity_data, data_file)
        logger.info(f"Intensity data saved to {data_file}")

    def _plot_intensity_graphs(
        self,
        time_flat: torch.Tensor,
        marked_times: dict[int, torch.Tensor],
        intensities_flat: torch.Tensor,
        intensities_at_marked_times: dict[int, torch.Tensor],
        num_mark: int,
        save_dir: str,
        plot: bool,
        save_plot: bool,
    ):
        """Generate and optionally save intensity plots."""
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)

        markers = ["o", "D", ",", "x", "+", "^", "v", "<", ">", "s", "p", "*"]

        for i in range(num_mark):
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            # Plot intensity curve
            ax.plot(
                time_flat.cpu().detach().numpy(),
                intensities_flat[:, i].cpu().detach().numpy(),
                color=f"C{i}",
                linewidth=2,
                label=f"Intensity Mark {i}",
            )

            # Plot observed events
            if len(marked_times[i]) > 0:
                ax.scatter(
                    marked_times[i].cpu().detach().numpy(),
                    intensities_at_marked_times[i].cpu().detach().numpy(),
                    s=30,
                    color=f"C{i}",
                    marker=markers[i % len(markers)],
                    label=f"Events Mark {i}",
                    alpha=0.8,
                )

            ax.set_title(f"Intensity for Mark {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plot:
                save_file = os.path.join(
                    save_dir, f"{self.__class__.__name__}_intensity_graph_mark_{i}.png"
                )
                plt.savefig(save_file, dpi=150, bbox_inches="tight")
                logger.info(f"Intensity graph for mark {i} saved to {save_file}")

            if plot:
                plt.show()
            else:
                plt.close()
