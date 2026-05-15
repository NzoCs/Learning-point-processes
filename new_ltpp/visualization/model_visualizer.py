"""ModelVisualizer — standalone visualization engine for TPP models.

Extracted from VisualizationMixin. Takes a model and a Simulator as dependencies.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Tuple
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from new_ltpp.utils import save_json, logger

if TYPE_CHECKING:
    from new_ltpp.models.model_protocol import ISimulableModel


class ModelVisualizer:
    """Standalone visualizer for TPP model intensity functions.

    Args:
        model: A model implementing ISimulableModel.
    """

    def __init__(self, model: "ISimulableModel") -> None:
        self._model = model
        self._simulator = model._simulator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intensity_graph(
        self,
        save_dir: str | Path,
        *,
        precision: int = 100,
        plot: bool = False,
        save_plot: bool = True,
        save_data: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Generate and visualize intensity curves for the model.

        Args:
            save_dir: Directory for saving outputs.
            precision: Number of interpolation points between events.
            plot: Whether to display the plots interactively.
            save_plot: Whether to save plots to disk.
            save_data: Whether to save intensity data as JSON.

        Returns:
            Tuple of (intensities_flat, time_flat, marked_times).
        """
        model = self._model

        time_seq, time_delta_seq, type_seq, seq_non_pad_mask = (
            self._get_simulation_data()
        )

        time_points, time_deltas_sample = self._generate_intensity_time_points(
            time_seq, time_delta_seq, precision
        )

        intensities = self._calculate_intensities(
            time_seq, time_delta_seq, type_seq, seq_non_pad_mask, time_deltas_sample
        )
        intensities_at_times = self._calculate_intensities(
            time_seq,
            time_delta_seq,
            type_seq,
            seq_non_pad_mask,
            time_delta_seq[:, 1:, None],
        )

        time_flat = time_points.view(-1)
        intensities_flat = intensities[0, ...].view(-1, model.num_event_types)

        marked_times, intensities_at_marked_times = (
            self._collect_marked_times_intensities(
                intensities_at_times,
                time_seq[:, 1:],
                type_seq[:, 1:],
                seq_non_pad_mask[:, 1:],
                model.num_event_types,
            )
        )

        if save_data:
            self._save_intensity_data(
                time_flat,
                intensities_flat,
                marked_times,
                save_dir,
                precision,
                model.num_event_types,
            )

        if plot or save_plot:
            self._plot_intensity_graphs(
                time_flat,
                marked_times,
                intensities_flat,
                intensities_at_marked_times,
                model.num_event_types,
                save_dir,
                plot,
                save_plot,
            )

        return intensities_flat, time_flat, marked_times

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_simulation_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        simul_result = self._simulator.simulate_from_scratch(num_sequences=1)
        return (
            simul_result.time_seqs,
            simul_result.time_delta_seqs,
            simul_result.type_seqs,
            simul_result.valid_event_mask,
        )

    def _generate_intensity_time_points(
        self,
        time_seq: torch.Tensor,
        time_delta_seq: torch.Tensor,
        precision: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._model.device
        ratios = (
            torch.linspace(0.0, 1.0, steps=precision, device=device)
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
        self,
        time_seqs: torch.Tensor,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_non_pad_mask: torch.Tensor,
        time_deltas_sample: torch.Tensor,
    ) -> torch.Tensor:
        model = self._model
        with torch.no_grad():
            intensities = model.compute_intensities_at_sample_dtimes(
                time_seqs=time_seqs,
                time_delta_seqs=time_delta_seqs,
                type_seqs=type_seqs,
                valid_event_mask=seq_non_pad_mask,
                sample_dtimes=time_deltas_sample,
                compute_last_step_only=False,
            )
        return intensities.detach().clone()

    def _collect_marked_times_intensities(
        self,
        intensities_at_times: torch.Tensor,
        time_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        seq_non_pad_mask: torch.Tensor,
        num_mark: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = self._model.device
        marked_times: List[torch.Tensor] = []
        intensities_at_marked: List[torch.Tensor] = []

        time_flat = time_seqs.squeeze(0)
        type_flat = type_seqs.squeeze(0)
        mask_flat = seq_non_pad_mask.squeeze(0)
        valid = (time_flat != 0) & mask_flat

        for i in range(num_mark):
            m = (type_flat == i) & valid
            if m.any():
                marked_times.append(time_flat[m])
                intensities_at_marked.append(intensities_at_times[0, :, 0, i][m])
            else:
                marked_times.append(torch.empty(0, device=device))
                intensities_at_marked.append(torch.empty(0, device=device))

        return marked_times, intensities_at_marked

    def _save_intensity_data(
        self,
        time_flat: torch.Tensor,
        intensities_flat: torch.Tensor,
        marked_times: List[torch.Tensor],
        save_dir: str | Path,
        precision: int,
        num_mark: int,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        model_name = self._model.__class__.__name__
        data = {
            "time_points": time_flat.cpu().detach().numpy().tolist(),
            "intensities": intensities_flat.cpu().detach().numpy().tolist(),
            "marked_times": {
                str(i): t.cpu().detach().numpy().tolist()
                for i, t in enumerate(marked_times)
            },
            "metadata": {
                "precision": precision,
                "num_event_types": num_mark,
                "model_type": model_name,
            },
        }
        data_file = os.path.join(save_dir, f"{model_name}_intensity_data.json")
        save_json(data, data_file)
        logger.info(f"ModelVisualizer: intensity data saved to {data_file}")

    def _plot_intensity_graphs(
        self,
        time_flat: torch.Tensor,
        marked_times: List[torch.Tensor],
        intensities_flat: torch.Tensor,
        intensities_at_marked_times: List[torch.Tensor],
        num_mark: int,
        save_dir: str | Path,
        plot: bool,
        save_plot: bool,
    ) -> None:
        model_name = self._model.__class__.__name__
        if save_plot:
            os.makedirs(save_dir, exist_ok=True)

        markers = ["o", "D", ",", "x", "+", "^", "v", "<", ">", "s", "p", "*"]

        for i in range(num_mark):
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(
                time_flat.cpu().detach().numpy(),
                intensities_flat[:, i].cpu().detach().numpy(),
                color=f"C{i}",
                linewidth=2,
                label=f"Intensity Mark {i}",
            )
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
                    save_dir, f"{model_name}_intensity_graph_mark_{i}.png"
                )
                plt.savefig(save_file, dpi=150, bbox_inches="tight")
                logger.info(f"ModelVisualizer: graph mark {i} saved to {save_file}")

            if plot:
                plt.show()
            else:
                plt.close()
