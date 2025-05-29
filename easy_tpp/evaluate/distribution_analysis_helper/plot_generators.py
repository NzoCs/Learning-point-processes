"""
Plot Generators for Temporal Point Process Analysis

This module provides specific plot generator implementations following
the Open/Closed Principle (OCP). Each generator handles a specific type of plot.

Author: Research Team
Date: 2024
"""

from easy_tpp.utils import logger
from .base_plot_generator import BasePlotGenerator
from .distribution_analyzer import DistributionAnalyzer
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class InterEventTimePlotGenerator(BasePlotGenerator):
    """Generates inter-event time distribution plots (OCP)."""
    
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_data = data['label_time_deltas']
        simulation_data = data['simulated_time_deltas']
        
        if len(label_data) == 0 or len(simulation_data) == 0:
            logger.warning("Insufficient inter-event time data for comparison")
            return
        
        datasets = [
            {
                'data': label_data,
                'label': 'Ground Truth', 
                'color': 'royalblue',
                'line_color': 'blue',
                'prefix': 'GT'
            },
            {
                'data': simulation_data,
                'label': 'Simulation',
                'color': 'crimson', 
                'line_color': 'red',
                'prefix': 'Sim'
            }
        ]
        
        DistributionAnalyzer.plot_density_comparison(
            datasets=datasets,
            output_path=output_path,
            title="Inter-Event Time Distribution Comparison",
            xlabel="Time Since Last Event"
        )
        
        # Generate QQ plot
        qq_path = output_path.replace('.png', '_qq.png')
        DistributionAnalyzer.create_qq_plot(
            label_data,
            simulation_data,
            "QQ Plot: Inter-Event Times (Ground Truth vs Simulation)",
            qq_path,
            log_scale=True
        )


class EventTypePlotGenerator(BasePlotGenerator):
    """Generates event type distribution plots (OCP)."""
    
    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
    
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_types = data['label_event_types']
        simulated_types = data['simulated_event_types']
        
        sns.set_theme(style="whitegrid")
        
        all_event_types = np.arange(self.num_event_types)
        
        label_types_int = label_types.astype(int)
        pred_types_int = simulated_types.astype(int)
        
        label_counts = np.bincount(label_types_int, minlength=self.num_event_types)
        pred_counts = np.bincount(pred_types_int, minlength=self.num_event_types)
        
        label_total = len(label_types)
        pred_total = len(simulated_types)
        
        label_probs = label_counts / label_total if label_total > 0 else np.zeros_like(label_counts)
        pred_probs = pred_counts / pred_total if pred_total > 0 else np.zeros_like(pred_counts)
        
        plt.figure(figsize=(10, 6))
        
        width = 0.35
        x = np.arange(len(all_event_types))
        
        plt.bar(x - width/2, label_probs, width, label='Ground Truth', 
              color='royalblue', alpha=0.7)
        plt.bar(x + width/2, pred_probs, width, label='Simulation', 
              color='crimson', alpha=0.7)
        
        plt.title('Comparison of Event Type Distributions', fontsize=14)
        plt.xlabel('Event Type', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(x, all_event_types)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics
        label_top3_indices = np.argsort(label_probs)[-3:][::-1]
        pred_top3_indices = np.argsort(pred_probs)[-3:][::-1]
        
        label_stats = "Ground Truth Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {label_probs[i]:.3f}" 
                                                        for i in label_top3_indices])
        pred_stats = "Simulation Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {pred_probs[i]:.3f}" 
                                                            for i in pred_top3_indices])
        
        plt.annotate(label_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.annotate(pred_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Event type distribution comparison plot saved to {output_path}")


class SequenceLengthPlotGenerator(BasePlotGenerator):
    """Generates sequence length distribution plots (OCP)."""
    
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_lengths = np.asarray(data['label_sequence_lengths'])
        simulated_lengths = np.asarray(data['simulated_sequence_lengths'])
        
        if len(label_lengths) == 0 or len(simulated_lengths) == 0:
            logger.warning("One or both sequence length arrays are empty. Skipping sequence length distribution plot.")
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        sns.histplot(label_lengths, label='Ground Truth', kde=False, 
                  stat='density', bins='auto', color='royalblue', alpha=0.6)
        sns.histplot(simulated_lengths, label='Simulation', kde=False, 
                  stat='density', bins='auto', color='crimson', alpha=0.6)
        
        # Calculate statistics
        label_mean = np.mean(label_lengths)
        label_median = np.median(label_lengths)
        label_std = np.std(label_lengths)
        
        pred_mean = np.mean(simulated_lengths)
        pred_median = np.median(simulated_lengths)
        pred_std = np.std(simulated_lengths)
        
        label_stats = (f"Ground Truth Stats:\n"
                     f"Mean: {label_mean:.2f}\n"
                     f"Median: {label_median:.2f}\n"
                     f"Std Dev: {label_std:.2f}")
        
        pred_stats = (f"Simulation Stats:\n"
                    f"Mean: {pred_mean:.2f}\n"
                    f"Median: {pred_median:.2f}\n"
                    f"Std Dev: {pred_std:.2f}")
        
        plt.annotate(label_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.annotate(pred_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.title('Comparison of Sequence Length Distributions', fontsize=14)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sequence length distribution comparison plot saved to {output_path}")
        
        # Add QQ plot
        qq_path = output_path.replace('.png', '_qq.png')
        DistributionAnalyzer.create_qq_plot(
            label_lengths, 
            simulated_lengths, 
            'QQ Plot: Sequence Lengths (Ground Truth vs Simulation)',
            qq_path,
            log_scale=False
        )


class CrossCorrelationPlotGenerator(BasePlotGenerator):
    """Generates cross-correlation plots (OCP)."""
    
    def generate_plot(self, data: Dict[str, Any], output_path: str) -> None:
        label_deltas = data['label_time_deltas']
        simulated_deltas = data['simulated_time_deltas']
        
        sns.set_theme(style="whitegrid")
        
        times_label = np.cumsum(label_deltas)
        times_pred = np.cumsum(simulated_deltas)
        
        if len(times_label) == 0 or len(times_pred) == 0:
            logger.warning("One or both time arrays are empty. Skipping cross-correlation plot.")
            return
        
        dt = 0.1
        T_max = max(np.max(times_label), np.max(times_pred)) + 1
        time_grid = np.arange(0, T_max, dt)

        N_label = self._build_count_process(times_label, time_grid)
        N_pred = self._build_count_process(times_pred, time_grid)

        h = r = int(2 / dt)
        max_lag = int(10 / dt)

        lags, label_corr = self._compute_cross_correlation(N_label, h, r, max_lag)
        _, pred_corr = self._compute_cross_correlation(N_pred, h, r, max_lag)

        plt.figure(figsize=(10, 6))
        plt.plot(lags * dt, label_corr, label='Ground Truth', linewidth=2, color='royalblue')
        plt.plot(lags * dt, pred_corr, label='Simulation', linewidth=2, linestyle='--', color='crimson')
        plt.axvline(0, color='gray', linestyle=':', linewidth=1)
        
        plt.title('Cross-Correlation of Counting Process Increments', fontsize=14)
        plt.xlabel('Time lag (x - t)', fontsize=12)
        plt.ylabel(r'$E[\Delta N(t,h) \cdot \Delta N(x,r)]$', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        
        annotation_text = (f"Window sizes: h = r = {2} time units\n"
                          f"Shows temporal dependencies in event occurrence")
        plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                   va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Cross-correlation comparison plot saved to {output_path}")
    
    @staticmethod
    def _build_count_process(event_times: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        return np.searchsorted(event_times, time_grid, side='right')

    @staticmethod
    def _compute_cross_correlation(N: np.ndarray, h: int, r: int, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimate E[(N(t+h)-N(t)) * (N(t+lag+r)-N(t+lag))] for various lags."""
        T = len(N)
        values = []
        lags = np.arange(-max_lag, max_lag + 1)
        
        for lag in lags:
            products = []
            for t in range(max(0, -lag), min(T - max(h, r) - abs(lag), T)):
                delta_t = N[t + h] - N[t]
                delta_x = N[t + lag + r] - N[t + lag]
                products.append(delta_t * delta_x)
            if products:
                values.append(np.mean(products))
            else:
                values.append(0.0)
        return lags, np.array(values)
