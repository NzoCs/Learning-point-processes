"""
Visualizer for Temporal Point Process Data.
"""

import os
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from new_ltpp.data.preprocess.data_loader import TPPDataModule
from new_ltpp.utils import logger
from new_ltpp.globals import OUTPUT_DIR


class Visualizer:
    """
    Visualizer for a single TPP dataset.

    Simplified and type-safe version focusing on a single dataset visualization.
    """

    def __init__(
        self,
        data_module: TPPDataModule,
        split: str = "test",
        save_dir: Optional[str] = None,
        max_events: int = 10**4,
    ):
        """
        Initialize the visualizer.

        Args:
            data_module: Data module containing the dataset
            split: Dataset split to use ('test', 'valid', or 'train')
            save_dir: Directory to save visualizations. If None, uses default logic.
            max_events: Maximum number of events to process for visualization
        """
        self.data_module = data_module
        dataset_name = data_module.data_config.dataset_id
        self.num_event_types = data_module.num_event_types
        self.split = split
        self.max_events = max_events

        # Validate split
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Split '{split}' is not valid. Choose from ['train', 'valid', 'test']"
            )

        # Determine save directory
        if save_dir:
            self.save_dir = save_dir
        else:
            # Default to OUTPUT_DIR/dataset_name/visualizations
            self.save_dir = str(OUTPUT_DIR / dataset_name / "visualizations")

        os.makedirs(self.save_dir, exist_ok=True)

        # Load and process data
        self.all_event_types: np.ndarray = np.array([])
        self.all_time_deltas: np.ndarray = np.array([])
        self.seq_lengths: np.ndarray = np.array([])
        self._load_data()

    def _load_data(self) -> None:
        """Load data from the data module and flatten it for analysis."""
        data_dir = self.data_module.data_config.get_data_dir(self.split)
        data_format = self.data_module.data_config.data_format

        logger.info(f"Loading data from {data_dir} for split {self.split}")
        raw_data = self.data_module.build_input(
            source_dir=data_dir, data_format=data_format, split=self.split
        )

        if "type_seqs" not in raw_data or "time_delta_seqs" not in raw_data:
            logger.warning("Data missing 'type_seqs' or 'time_delta_seqs'.")
            return

        type_seqs = raw_data["type_seqs"]
        time_seqs = raw_data["time_delta_seqs"]

        all_event_types_list: List[float] = []
        all_time_deltas_list: List[float] = []
        seq_lengths_list: List[int] = []

        processed_events = 0

        for type_seq, time_seq in zip(type_seqs, time_seqs):
            seq_lengths_list.append(len(type_seq))

            # Add events until we reach max_events
            n_events = len(type_seq)
            if processed_events < self.max_events:
                remaining = self.max_events - processed_events

                # Take all or a slice
                take = min(n_events, remaining)

                all_event_types_list.extend(type_seq[:take])
                all_time_deltas_list.extend(time_seq[:take])

                processed_events += take

        self.all_event_types = np.array(all_event_types_list)
        self.all_time_deltas = np.array(all_time_deltas_list)
        self.seq_lengths = np.array(seq_lengths_list)

        logger.info(
            f"Loaded {len(self.all_event_types)} events from {len(self.seq_lengths)} sequences."
        )

    def plot_inter_event_times(
        self,
        filename: str = "inter_event_time_dist.png",
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot the distribution of inter-event times.

        Args:
            filename: Output filename
            show: Whether to display the plot using plt.show()
        """
        if len(self.all_time_deltas) == 0:
            logger.warning("No time deltas to plot.")
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        times_data = self.all_time_deltas

        # Plot Histogram on Log scale
        # Using matplotlib hist with log=True for y-axis log scale
        plt.hist(
            times_data,
            bins=50,
            log=True,
            color="royalblue",
            alpha=0.7,
            label=f"Data ({self.split})",
        )

        # Calculate statistics
        mean = np.mean(times_data)
        median = np.median(times_data)
        std = np.std(times_data)

        stats_text = f"Mean: {mean:.4f}\nMedian: {median:.4f}\nStd Dev: {std:.4f}"

        # Add stats box
        plt.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        )

        plt.title("Inter-Event Time Distribution", fontsize=14)
        plt.xlabel("Time Since Last Event", fontsize=12)
        plt.ylabel("Frequency (Log Scale)", fontsize=12)
        plt.tight_layout()

        self._save_and_show(filename, save, show)

    def plot_event_types(
        self,
        filename: str = "event_type_dist.png",
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot the distribution of event types.

        Args:
            filename: Output filename
            show: Whether to display the plot
        """
        if len(self.all_event_types) == 0:
            logger.warning("No event types to plot.")
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Calculate probabilities
        types_int = self.all_event_types.astype(int)
        counts = np.bincount(types_int, minlength=self.num_event_types)
        total_counts = counts.sum()
        probs = (
            counts / total_counts
            if total_counts > 0
            else np.zeros_like(counts, dtype=float)
        )

        x = np.arange(self.num_event_types)

        plt.bar(x, probs, color="royalblue", alpha=0.7, label=f"Data ({self.split})")

        # Top 3 types statistics
        top_indices = np.argsort(probs)[-3:][::-1]
        stats_text = "Top Types:\n" + "\n".join(
            [f"Type {i}: {probs[i]:.3f}" for i in top_indices]
        )

        plt.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        )

        plt.title("Event Type Distribution", fontsize=14)
        plt.xlabel("Event Type", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.xticks(x)
        plt.tight_layout()

        self._save_and_show(filename, save, show)

    def plot_sequence_lengths(
        self,
        filename: str = "sequence_length_dist.png",
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot the distribution of sequence lengths.

        Args:
            filename: Output filename
            show: Whether to display the plot
        """
        if len(self.seq_lengths) == 0:
            logger.warning("No sequences to plot.")
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Determine decent bin width
        std_val = np.std(self.seq_lengths)
        binwidth = max(1, int(std_val / 2))

        sns.histplot(
            self.seq_lengths,
            stat="density",
            binwidth=binwidth,
            color="royalblue",
            alpha=0.6,
            label=f"Data ({self.split})",
        )

        # Statistics
        mean = np.mean(self.seq_lengths)
        median = np.median(self.seq_lengths)
        std = np.std(self.seq_lengths)

        stats_text = f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std:.2f}"

        plt.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        )

        plt.title("Sequence Length Distribution", fontsize=14)
        plt.xlabel("Sequence Length", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.tight_layout()

        self._save_and_show(filename, save, show)

    def _save_and_show(self, filename: str, save: bool, show: bool) -> None:
        """Helper to save and optionally show the plot."""
        if save:
            filepath = os.path.join(self.save_dir, filename)
            try:
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Plot saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot to {filepath}: {e}")

        if show:
            try:
                plt.show()
            except Exception:
                # Ignore errors in headless environments
                pass
        plt.close()
