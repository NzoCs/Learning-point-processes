"""
Plot Generators for Temporal Point Process Analysis
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from new_ltpp.utils import logger

from .base_plot_generator import BasePlotGenerator


class InterEventTimePlotGenerator(BasePlotGenerator):
    """Generates inter-event time distribution plots (OCP)."""

    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_hist = data["label_time_deltas"]  # Histogram counts
        simulation_hist = data["simulated_time_deltas"]  # Histogram counts
        bin_edges = data["time_bin_edges"]

        if label_hist.sum() == 0 or simulation_hist.sum() == 0:
            logger.warning("Insufficient inter-event time data for comparison")
            return

        sns.set_theme(style="whitegrid")

        # Compute bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Normalize to density
        label_density = label_hist / (label_hist.sum() * bin_widths)
        sim_density = simulation_hist / (simulation_hist.sum() * bin_widths)

        # Histogram plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_centers,
            label_density,
            width=bin_widths,
            alpha=0.6,
            label="Ground Truth",
            color="royalblue",
        )
        plt.bar(
            bin_centers,
            sim_density,
            width=bin_widths,
            alpha=0.6,
            label="Simulation",
            color="crimson",
        )
        plt.title("Inter-Event Time Distribution Comparison", fontsize=14)
        plt.xlabel("Time Since Last Event", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_output_path = output_path.replace(".png", "_histogram.png")
        plt.savefig(hist_output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Inter-event time histogram saved to {hist_output_path}")

        # QQ-plot
        plt.figure(figsize=(10, 6))

        # Reconstruct approximate quantiles from histogram using cumulative distribution
        cumsum_label = np.cumsum(label_hist)
        cumsum_label = cumsum_label / cumsum_label[-1]  # Normalize to [0, 1]

        cumsum_sim = np.cumsum(simulation_hist)
        cumsum_sim = cumsum_sim / cumsum_sim[-1]  # Normalize to [0, 1]

        # Sample quantiles at regular intervals
        quantile_levels = np.linspace(0.01, 0.99, 100)
        qq_label = np.interp(quantile_levels, cumsum_label, bin_centers)
        qq_sim = np.interp(quantile_levels, cumsum_sim, bin_centers)

        plt.scatter(qq_label, qq_sim, alpha=0.5, s=10, color="navy")

        # Reference line
        min_val = min(qq_label.min(), qq_sim.min())
        max_val = max(qq_label.max(), qq_sim.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect match",
        )

        plt.title("Q-Q Plot: Inter-Event Time Comparison", fontsize=14)
        plt.xlabel("Ground Truth Quantiles", fontsize=12)
        plt.ylabel("Simulation Quantiles", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        qq_output_path = output_path.replace(".png", "_qq_plot.png")
        plt.savefig(qq_output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Inter-event time Q-Q plot saved to {qq_output_path}")


class EventTypePlotGenerator(BasePlotGenerator):
    """Generates event type distribution plots (OCP)."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types

    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_types = data["label_event_types"]
        simulated_types = data["simulated_event_types"]

        sns.set_theme(style="whitegrid")

        all_event_types = np.arange(self.num_event_types)

        label_types_int = label_types.astype(int)
        pred_types_int = simulated_types.astype(int)

        label_counts = np.bincount(label_types_int, minlength=self.num_event_types)
        pred_counts = np.bincount(pred_types_int, minlength=self.num_event_types)

        label_total = len(label_types)
        pred_total = len(simulated_types)

        label_probs = (
            label_counts / label_total
            if label_total > 0
            else np.zeros_like(label_counts)
        )
        pred_probs = (
            pred_counts / pred_total if pred_total > 0 else np.zeros_like(pred_counts)
        )

        plt.figure(figsize=(10, 6))

        width = 0.35
        x = np.arange(len(all_event_types))

        plt.bar(
            x - width / 2,
            label_probs,
            width,
            label="Ground Truth",
            color="royalblue",
            alpha=0.7,
        )
        plt.bar(
            x + width / 2,
            pred_probs,
            width,
            label="Simulation",
            color="crimson",
            alpha=0.7,
        )

        plt.title("Comparison of Event Type Distributions", fontsize=14)
        plt.xlabel("Event Type", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.xticks(x, [str(t) for t in all_event_types])
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Event type distribution comparison plot saved to {output_path}")


class SequenceLengthPlotGenerator(BasePlotGenerator):
    """Generates sequence length distribution plots (OCP)."""

    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:

        label_lengths = np.asarray(data["label_sequence_lengths"])
        simulated_lengths = np.asarray(data["simulated_sequence_lengths"])

        if len(label_lengths) == 0 or len(simulated_lengths) == 0:
            logger.warning(
                "One or both sequence length arrays are empty. Skipping sequence length distribution plot."
            )
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Calculer des bins communs pour les deux distributions
        all_lengths = np.concatenate([label_lengths, simulated_lengths])
        common_bins = np.histogram_bin_edges(all_lengths, bins="auto")

        sns.histplot(
            label_lengths,
            label="Ground Truth",
            kde=False,
            stat="count",
            bins=common_bins,
            color="royalblue",
            alpha=0.6,
        )
        sns.histplot(
            simulated_lengths,
            label="Simulation",
            kde=False,
            stat="count",
            bins=common_bins,
            color="crimson",
            alpha=0.6,
        )

        plt.title("Comparison of Sequence Length Distributions", fontsize=14)
        plt.xlabel("Sequence Length", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            f"Sequence length distribution comparison plot saved to {output_path}"
        )


class AutocorrelationPlotGenerator(BasePlotGenerator):
    """Generates autocorrelation function (ACF) plots similar to statsmodels."""

    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        acf_gt = data["acf_gt_mean"]
        acf_sim = data["acf_sim_mean"]

        if len(acf_gt) == 0 or len(acf_sim) == 0:
            logger.warning("ACF data is empty. Skipping autocorrelation plot.")
            return

        sns.set_theme(style="whitegrid")

        lags = np.arange(len(acf_gt))

        plt.figure(figsize=(12, 8))

        # Plot ground truth ACF with stems (like statsmodels)
        markerline_gt, stemlines_gt, baseline_gt = plt.stem(
            lags,
            acf_gt,
            linefmt="royalblue",
            markerfmt="bo",
            basefmt="gray",
            label="Ground Truth",
        )
        plt.setp(stemlines_gt, linewidth=1.5)
        plt.setp(markerline_gt, markersize=6, alpha=0.8)

        # Plot simulation ACF with stems
        markerline_sim, stemlines_sim, baseline_sim = plt.stem(
            lags + 0.1,
            acf_sim,  # Slight offset to avoid overlap
            linefmt="crimson",
            markerfmt="rs",
            basefmt="gray",
            label="Simulation",
        )
        plt.setp(stemlines_sim, linewidth=1.5, linestyle="--")
        plt.setp(markerline_sim, markersize=6, alpha=0.8)

        # Confidence bands (approximate, like statsmodels)
        # For white noise, 95% confidence interval is approximately ±1.96/sqrt(n)
        # But here we'll use a simple approximation
        n_samples = 100  # Approximate sample size for confidence calculation
        conf_interval = 1.96 / np.sqrt(n_samples)

        plt.axhline(
            y=conf_interval, color="gray", linestyle="--", alpha=0.7, linewidth=1
        )
        plt.axhline(
            y=-conf_interval, color="gray", linestyle="--", alpha=0.7, linewidth=1
        )
        plt.axhline(y=0, color="black", linewidth=1)

        # Fill confidence region
        plt.fill_between(
            lags,
            -conf_interval,
            conf_interval,
            color="gray",
            alpha=0.1,
            label="95% Confidence Interval",
        )

        plt.title(
            "Autocorrelation Function (ACF) Comparison", fontsize=16, fontweight="bold"
        )
        plt.xlabel("Lag", fontsize=14)
        plt.ylabel("Autocorrelation", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        # Set x-ticks to show all lags
        plt.xticks(lags)

        # Adjust y-limits to show confidence bands
        y_min = min(np.min(acf_gt), np.min(acf_sim), -conf_interval * 1.2)
        y_max = max(np.max(acf_gt), np.max(acf_sim), conf_interval * 1.2)
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Autocorrelation comparison plot saved to {output_path}")
