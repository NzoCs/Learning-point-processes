"""
Distribution Analysis Utilities for Temporal Point Processes
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from new_ltpp.utils import logger


class DistributionAnalyzer:
    """Utility class for statistical analysis and visualization (SRP)."""

    @staticmethod
    def plot_density_comparison(
        datasets: List[Dict[str, Any]],
        output_path: str,
        title: str = "Density Comparison",
        xlabel: str = "Value",
        figsize: Tuple[int, int] = (10, 6),
        n_bins: int = 50,
        alpha: float = 0.4,
    ) -> None:
        """
        Create density comparison plots between multiple datasets.

        Args:
            datasets: List of dataset dictionaries containing:
                - data: numpy array of values
                - label: string label for legend
                - color: color for histogram
            output_path: Full path where to save the plot
            title: Plot title
            xlabel: X-axis label
            figsize: Figure size as (width, height)
            n_bins: Number of bins for histogram
            alpha: Transparency level for histograms

        Raises:
            ValueError: If datasets list is empty or contains invalid data
        """
        if not datasets:
            raise ValueError("Datasets list cannot be empty")

        # Validate dataset format
        required_keys = {"data", "label", "color"}
        for i, dataset in enumerate(datasets):
            missing_keys = required_keys - set(dataset.keys())
            if missing_keys:
                raise ValueError(f"Dataset {i} missing required keys: {missing_keys}")

        fig = None
        try:
            # Setup plotting environment
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=figsize)

            # Create unified bins for fair comparison across datasets
            all_data = np.concatenate(
                [ds["data"] for ds in datasets if len(ds["data"]) > 0]
            )
            if len(all_data) == 0:
                logger.warning("All datasets are empty, skipping plot generation")
                return

            bins = np.histogram_bin_edges(all_data, bins=n_bins)

            # Process each dataset
            for dataset in datasets:
                if len(dataset["data"]) == 0:
                    logger.warning(f"Empty dataset for {dataset['label']}, skipping")
                    continue

                # Generate histogram with density normalization
                sns.histplot(
                    dataset["data"],
                    bins=bins,
                    stat="density",
                    element="step",
                    fill=True,
                    alpha=alpha,
                    label=dataset["label"],
                    ax=ax,
                    color=dataset["color"],
                )

                # Add statistical summary
                DistributionAnalyzer._add_statistical_summary(
                    dataset, ax, len(datasets)
                )

            # Configure plot appearance
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel("Density (log scale)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(loc="center right")
            ax.grid(True, alpha=0.3)

            # Save plot with high quality
            fig.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Density comparison plot successfully saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create density comparison plot: {str(e)}")
            if fig is not None:
                plt.close(fig)
            raise

    @staticmethod
    def _add_statistical_summary(
        dataset: Dict[str, Any], ax, total_datasets: int
    ) -> None:
        """
        Add statistical summary text box to the plot.

        Args:
            dataset: Dataset information dictionary
            ax: Matplotlib axes object
            total_datasets: Total number of datasets being compared
        """
        try:
            data = dataset["data"]
            if len(data) == 0:
                return

            # Calculate comprehensive statistics
            stats_text = (
                f"{dataset['label']}:\n"
                f"Mean: {np.mean(data):.4f}\n"
                f"Median: {np.median(data):.4f}\n"
                f"Std Dev: {np.std(data):.4f}\n"
                f"Sample size: {len(data):,}"
            )

            # Determine optimal text positioning
            positions = [
                {"xy": (0.02, 0.95), "ha": "left"},  # Left side for ground truth
                {"xy": (0.98, 0.95), "ha": "right"},  # Right side for simulation
            ]

            # Smart positioning based on dataset type
            is_ground_truth = any(
                keyword in dataset["label"].lower()
                for keyword in ["truth", "ground", "real", "actual"]
            )
            position_index = 0 if is_ground_truth else min(1, total_datasets - 1)

            if position_index < len(positions):
                pos = positions[position_index]
                ax.text(
                    pos["xy"][0],
                    pos["xy"][1],
                    stats_text,
                    transform=ax.transAxes,
                    va="top",
                    ha=pos["ha"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=10,
                )
        except Exception as e:
            logger.warning(
                f"Failed to add statistical summary for {dataset['label']}: {str(e)}"
            )

    @staticmethod
    def create_qq_plot(
        reference_data: np.ndarray,
        comparison_data: np.ndarray,
        title: str,
        save_path: str,
        log_scale: bool = False,
    ) -> None:
        """
        Create and save a quantile-quantile (QQ) plot for distribution comparison.

        Args:
            reference_data: Reference dataset (typically ground truth)
            comparison_data: Comparison dataset (typically predictions/simulations)
            title: Plot title
            save_path: Full path to save the plot
            log_scale: Whether to use logarithmic scaling

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Input validation and conversion
            reference_data = np.asarray(reference_data)
            comparison_data = np.asarray(comparison_data)

            # Filter data for log scale if needed
            if log_scale:
                reference_data = reference_data[reference_data > 0]
                comparison_data = comparison_data[comparison_data > 0]

            # Validate sufficient data
            if len(reference_data) == 0 or len(comparison_data) == 0:
                raise ValueError("Insufficient data after filtering for QQ plot")

            # Optimize memory usage by sorting in-place
            reference_data.sort()
            comparison_data.sort()

            # Determine quantiles to compute
            n_quantiles = min(len(reference_data), len(comparison_data))
            if n_quantiles < 10:
                logger.warning(f"Limited data points for QQ plot: {n_quantiles}")

            # Generate evenly spaced quantiles (avoiding extremes)
            quantiles = np.linspace(0.01, 0.99, min(100, n_quantiles))

            # Compute quantiles efficiently
            ref_quantiles = np.quantile(reference_data, quantiles)
            comp_quantiles = np.quantile(comparison_data, quantiles)

            # Create QQ plot
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(8, 8))

            if log_scale:
                plt.loglog(
                    ref_quantiles,
                    comp_quantiles,
                    "o",
                    markersize=4,
                    color="royalblue",
                    alpha=0.7,
                )

                # Reference line for log-log plot
                min_val = max(min(ref_quantiles.min(), comp_quantiles.min()), 1e-10)
                max_val = max(ref_quantiles.max(), comp_quantiles.max())
                ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
                plt.loglog(ref_line, ref_line, "r--", alpha=0.7, linewidth=2)
            else:
                plt.plot(
                    ref_quantiles,
                    comp_quantiles,
                    "o",
                    markersize=4,
                    color="royalblue",
                    alpha=0.7,
                )

                # Reference line for linear plot
                min_val = min(ref_quantiles.min(), comp_quantiles.min())
                max_val = max(ref_quantiles.max(), comp_quantiles.max())
                ref_line = np.linspace(min_val, max_val, 100)
                plt.plot(ref_line, ref_line, "r--", alpha=0.7, linewidth=2)

            # Configure plot appearance
            plt.grid(True, alpha=0.3)
            plt.title(title, fontsize=14)
            plt.xlabel("Reference Quantiles", fontsize=12)
            plt.ylabel("Comparison Quantiles", fontsize=12)

            # Add interpretive annotation
            annotation_text = (
                "Points along diagonal indicate similar distributions.\n"
                "Systematic deviations suggest distributional differences."
            )
            plt.annotate(
                annotation_text,
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
            )

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"QQ plot successfully saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to create QQ plot: {str(e)}")
            plt.close("all")  # Clean up any open figures
            raise
