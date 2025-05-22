from easy_tpp.utils import logger

from typing import Dict, List, Union, Optional
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class NewDistribComparator:
    
    def __init__(
            self,
            label_data_loader,
            simulation: List[Dict],
            num_event_types: int,
            output_dir: str,
            dataset_size: int = 10**5
            ):
        """
        Initialize the comparator for simulation evaluation.
        
        Args:
            label_data_loader: DataLoader containing the ground truth data
            simulation: List of dictionaries containing simulated sequences
            num_event_types: Number of event types in the dataset
            output_dir: Directory to save output plots
            dataset_size: Maximum number of events to use for comparison
        """
        self.output_dir = output_dir
        self.num_event_types = num_event_types
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data from label_data_loader
        self.label_time_deltas = []
        self.label_event_types = []
        self.label_sequence_lengths = []
        
        # Process label data
        logger.info("Extracting label data from dataloader...")
        for batch in label_data_loader:
            # Get the relevant tensors from the batch
            if isinstance(batch, dict):
                time_delta_seqs = batch.get('time_delta_seqs', None)
                type_seqs = batch.get('type_seqs', None)
                attention_mask = batch.get('attention_mask', None)
            else:
                # Assuming it's a tuple or list with a specific order
                batch_values = list(batch.values()) if hasattr(batch, 'values') else batch
                if len(batch_values) >= 5:
                    _, time_delta_seqs, type_seqs, attention_mask, _ = batch_values
                else:
                    logger.warning(f"Batch format not recognized: {type(batch)}")
                    continue
            
            # Process each sequence in the batch
            for i in range(len(time_delta_seqs)):
                mask = attention_mask[i] if attention_mask is not None else torch.ones_like(time_delta_seqs[i]).bool()
                valid_indices = mask.bool()
                
                # Extract valid time deltas and event types
                time_deltas = time_delta_seqs[i][valid_indices].cpu().numpy()
                event_types = type_seqs[i][valid_indices].cpu().numpy()
                
                # Store sequence length
                self.label_sequence_lengths.append(len(time_deltas))
                
                # Add to our lists
                self.label_time_deltas.extend(time_deltas)
                self.label_event_types.extend(event_types)
                
                # Check if we've collected enough data
                if len(self.label_time_deltas) >= dataset_size:
                    break
            
            if len(self.label_time_deltas) >= dataset_size:
                break
        
        # Process simulation data
        self.simulated_time_deltas = []
        self.simulated_event_types = []
        self.simulated_sequence_lengths = []
        
        logger.info("Processing simulation data...")
        for seq in simulation:
            if 'time_delta_seq' in seq and 'event_seq' in seq:
                # Extract time deltas and event types
                time_deltas = seq['time_delta_seq'].cpu().numpy()
                event_types = seq['event_seq'].cpu().numpy()
            else:
                continue
                
            # Store sequence length
            self.simulated_sequence_lengths.append(len(time_deltas))
            
            # Add to our lists
            self.simulated_time_deltas.extend(time_deltas)
            self.simulated_event_types.extend(event_types)
            
            # Check if we've collected enough data
            if len(self.simulated_time_deltas) >= dataset_size:
                break
        
        # Convert to numpy arrays for faster processing
        self.label_time_deltas = np.array(self.label_time_deltas[:dataset_size])
        self.label_event_types = np.array(self.label_event_types[:dataset_size])
        self.simulated_time_deltas = np.array(self.simulated_time_deltas[:dataset_size])
        self.simulated_event_types = np.array(self.simulated_event_types[:dataset_size])
        
        logger.info(f"Collected {len(self.label_time_deltas)} label events and {len(self.simulated_time_deltas)} simulated events for comparison")
        
        # Generate all plots automatically upon initialization
        self.run_evaluation()
    
    def plot_inter_event_time_distribution(self, filename: str = "comparison_inter_event_time_dist.png"):
        """
        Plots and saves the superimposed distribution of inter-event times.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Main distribution plot
        plt.figure(figsize=(10, 6))
        
        # Use Seaborn's histplot or kdeplot for better aesthetics
        if len(self.label_time_deltas) > 0 and len(self.simulated_time_deltas) > 0:
            # Create histograms for both datasets
            label_hist, label_bin_edges = np.histogram(self.label_time_deltas, bins=50)
            pred_hist, pred_bin_edges = np.histogram(self.simulated_time_deltas, bins=50)
            
            # Get bin centers for plotting
            label_bin_centers = (label_bin_edges[:-1] + label_bin_edges[1:]) / 2
            pred_bin_centers = (pred_bin_edges[:-1] + pred_bin_edges[1:]) / 2
            
            # Filter out bins with frequency <= 20
            label_mask = label_hist > 20
            pred_mask = pred_hist > 20
            
            # Filter the histogram data and bin centers
            filtered_label_hist = label_hist[label_mask]
            filtered_label_bin_centers = label_bin_centers[label_mask]
            filtered_pred_hist = pred_hist[pred_mask]
            filtered_pred_bin_centers = pred_bin_centers[pred_mask]
            
            # Plot histograms - we'll use bar plots with filtered data instead of histplot
            plt.bar(filtered_label_bin_centers, filtered_label_hist, width=(label_bin_edges[1] - label_bin_edges[0]) * 0.8,
                  label='Ground Truth', alpha=0.6, color='royalblue')
            plt.bar(filtered_pred_bin_centers, filtered_pred_hist, width=(pred_bin_edges[1] - pred_bin_edges[0]) * 0.8,
                  label='Simulation', alpha=0.6, color='crimson')
            
            plt.yscale('log')
            
            # Calculate linear regression for filtered label data
            if len(filtered_label_hist) > 1:  # Need at least 2 points for regression
                label_X = filtered_label_bin_centers.reshape(-1, 1)
                label_y = np.log10(filtered_label_hist)  # Log scale since y-axis is log
                label_reg = np.polyfit(label_X.flatten(), label_y, 1)
                label_slope, label_intercept = label_reg
                
                # Generate points for the regression line
                label_x_line = np.linspace(min(filtered_label_bin_centers), max(filtered_label_bin_centers), 100)
                label_y_line = 10 ** (label_slope * label_x_line + label_intercept)
                
                # Plot the regression line
                plt.plot(label_x_line, label_y_line, '--', color='blue', 
                       label=f'Truth slope: {label_slope:.4f}')
            
            # Calculate linear regression for filtered prediction data
            if len(filtered_pred_hist) > 1:  # Need at least 2 points for regression
                pred_X = filtered_pred_bin_centers.reshape(-1, 1)
                pred_y = np.log10(filtered_pred_hist)  # Log scale since y-axis is log
                pred_reg = np.polyfit(pred_X.flatten(), pred_y, 1)
                pred_slope, pred_intercept = pred_reg
                
                # Generate points for the regression line
                pred_x_line = np.linspace(min(filtered_pred_bin_centers), max(filtered_pred_bin_centers), 100)
                pred_y_line = 10 ** (pred_slope * pred_x_line + pred_intercept)
                
                # Plot the regression line
                plt.plot(pred_x_line, pred_y_line, '--', color='red', 
                       label=f'Simul slope: {pred_slope:.4f}')
            
            # Calculate statistics using vectorized operations
            label_mean = np.mean(self.label_time_deltas)
            label_median = np.median(self.label_time_deltas)
            label_std = np.std(self.label_time_deltas)
            
            pred_mean = np.mean(self.simulated_time_deltas)
            pred_median = np.median(self.simulated_time_deltas)
            pred_std = np.std(self.simulated_time_deltas)
            
            # Add statistics to the plot as annotations
            label_stats = (f"Ground Truth Stats:\n"
                         f"Mean: {label_mean:.4f}\n"
                         f"Median: {label_median:.4f}\n"
                         f"Std Dev: {label_std:.4f}")
            
            pred_stats = (f"Simulation Stats:\n"
                        f"Mean: {pred_mean:.4f}\n"
                        f"Median: {pred_median:.4f}\n"
                        f"Std Dev: {pred_std:.4f}")
            
            # Position text for label stats at top left
            plt.annotate(label_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                      va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            # Position text for prediction stats at top right
            plt.annotate(pred_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                      va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        else:
            logger.warning("One or both of the time delta arrays are empty. Skipping plot generation.")
            return

        plt.title('Comparison of Inter-Event Time Distributions (Log Scale)')
        plt.xlabel('Time Since Last Event (Log Scale)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Inter-event time distribution comparison plot saved to {filepath}")
        
        # Add QQ plot for inter-event times
        self._create_qq_plot(
            self.label_time_deltas, 
            self.simulated_time_deltas, 
            'QQ Plot: Inter-Event Times (Ground Truth vs Simulation)',
            os.path.join(self.output_dir, "qq_inter_event_times.png"),
            log_scale=True
        )

    def _create_qq_plot(self, label_data: np.ndarray, pred_data: np.ndarray, title: str, save_path: str, log_scale: bool = False):
        """
        Creates and saves a QQ plot comparing label and prediction distributions.
        
        Args:
            label_data: Data points from the label dataset
            pred_data: Data points from the prediction dataset
            title: Title for the plot
            save_path: Path to save the plot
            log_scale: Whether to use log scale for the axes
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")
        
        # Convert inputs to numpy arrays if they aren't already
        label_data = np.asarray(label_data)
        pred_data = np.asarray(pred_data)
        
        # Filter out non-positive values if using log scale - use vectorized operations
        if log_scale:
            label_data = label_data[label_data > 0]
            pred_data = pred_data[pred_data > 0]
        
        # Skip if empty arrays
        if len(label_data) == 0 or len(pred_data) == 0:
            logger.warning(f"Empty arrays after filtering: label_data={len(label_data)}, pred_data={len(pred_data)}")
            return
            
        # Get quantiles using vectorized operations
        # Sort the arrays in-place for better memory efficiency
        label_data.sort()
        pred_data.sort()
        
        # Determine number of quantiles to use (minimum to avoid extrapolation)
        n_quantiles = min(len(label_data), len(pred_data))
        
        if n_quantiles < 10:
            logger.warning(f"Not enough data points for QQ plot: {n_quantiles} points")
            return
        
        # Create evenly spaced quantiles, avoiding 0 and 1 to prevent infinity issues with some distributions
        quantiles = np.linspace(0.01, 0.99, min(100, n_quantiles))
        
        # Compute quantiles efficiently using vectorized numpy functions
        label_quantiles = np.quantile(label_data, quantiles)
        pred_quantiles = np.quantile(pred_data, quantiles)
        
        # Create QQ plot
        plt.figure(figsize=(8, 8))
        
        # Plot the quantiles
        if log_scale:
            plt.loglog(label_quantiles, pred_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x) in log-log space
            min_val = max(min(label_quantiles.min(), pred_quantiles.min()), 1e-10)  # Avoid log(0)
            max_val = max(label_quantiles.max(), pred_quantiles.max())
            ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            plt.loglog(ref_line, ref_line, 'r--', alpha=0.7)
        else:
            plt.plot(label_quantiles, pred_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x)
            min_val = min(label_quantiles.min(), pred_quantiles.min())
            max_val = max(label_quantiles.max(), pred_quantiles.max())
            ref_line = np.linspace(min_val, max_val, 100)
            plt.plot(ref_line, ref_line, 'r--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.title(title, fontsize=14)
        plt.xlabel('Ground Truth Quantiles', fontsize=12)
        plt.ylabel('Simulation Quantiles', fontsize=12)
        
        # Add annotation explaining interpretation with a box
        annotation_text = ("Points along reference line indicate similar distributions.\n"
                         "Deviations suggest distributional differences.")
        plt.annotate(annotation_text, xy=(0.05, 0.05), xycoords='axes fraction',
                   va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"QQ plot saved to {save_path}")

    def plot_event_type_distribution(self, filename: str = "comparison_event_type_dist.png"):
        """
        Plots and saves the superimposed distribution of event types.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Calculate the probability distribution using vectorized operations
        # Get all possible event types
        all_event_types = np.arange(self.num_event_types)
        
        # Count occurrences using numpy's bincount (faster than Counter for large arrays)
        # Ensure the array contains only integers
        label_types_int = self.label_event_types.astype(int)
        pred_types_int = self.simulated_event_types.astype(int)
        
        # Use bincount with minlength to ensure all event types are counted
        label_counts = np.bincount(label_types_int, minlength=self.num_event_types)
        pred_counts = np.bincount(pred_types_int, minlength=self.num_event_types)
        
        # Calculate normalized counts (probabilities) using numpy's division
        label_total = len(self.label_event_types)
        pred_total = len(self.simulated_event_types)
        
        label_probs = label_counts / label_total if label_total > 0 else np.zeros_like(label_counts)
        pred_probs = pred_counts / pred_total if pred_total > 0 else np.zeros_like(pred_counts)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Width of bars - slightly offset for better visibility
        width = 0.35
        x = np.arange(len(all_event_types))
        
        # Plot bars side by side
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
        
        # Add statistics for top event types as annotations
        # Use numpy's argsort to get indices of top event types
        label_top3_indices = np.argsort(label_probs)[-3:][::-1]  # Get top 3 in descending order
        pred_top3_indices = np.argsort(pred_probs)[-3:][::-1]
        
        # Create formatted statistics strings
        label_stats = "Ground Truth Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {label_probs[i]:.3f}" 
                                                        for i in label_top3_indices])
        pred_stats = "Simulation Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {pred_probs[i]:.3f}" 
                                                            for i in pred_top3_indices])
        
        # Position text on left and right sides
        plt.annotate(label_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.annotate(pred_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Event type distribution comparison plot saved to {filepath}")

    def plot_sequence_length_distribution(self, filename: str = "comparison_sequence_length_dist.png"):
        """
        Plots and saves the superimposed distribution of sequence lengths.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Convert to numpy arrays if not already for faster operations
        label_lengths = np.asarray(self.label_sequence_lengths)
        pred_lengths = np.asarray(self.simulated_sequence_lengths)
        
        # Check if the arrays are empty
        if len(label_lengths) == 0 or len(pred_lengths) == 0:
            logger.warning("One or both sequence length arrays are empty. Skipping sequence length distribution plot.")
            return

        plt.figure(figsize=(10, 6))
        
        # Use Seaborn's histplot with optimized parameters
        sns.histplot(label_lengths, label='Ground Truth', kde=True, 
                  stat='density', bins=50, color='royalblue', alpha=0.6)
        sns.histplot(pred_lengths, label='Simulation', kde=True, 
                  stat='density', bins=50, color='crimson', alpha=0.6)
        
        # Calculate statistics using vectorized operations
        label_mean = np.mean(label_lengths)
        label_median = np.median(label_lengths)
        label_std = np.std(label_lengths)
        
        pred_mean = np.mean(pred_lengths)
        pred_median = np.median(pred_lengths)
        pred_std = np.std(pred_lengths)
        
        # Add statistics to the plot as annotations
        label_stats = (f"Ground Truth Stats:\n"
                     f"Mean: {label_mean:.2f}\n"
                     f"Median: {label_median:.2f}\n"
                     f"Std Dev: {label_std:.2f}")
        
        pred_stats = (f"Simulation Stats:\n"
                    f"Mean: {pred_mean:.2f}\n"
                    f"Median: {pred_median:.2f}\n"
                    f"Std Dev: {pred_std:.2f}")
        
        # Position text for label stats at top left
        plt.annotate(label_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        # Position text for prediction stats at top right
        plt.annotate(pred_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.title('Comparison of Sequence Length Distributions', fontsize=14)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sequence length distribution comparison plot saved to {filepath}")
        
        # Add QQ plot for sequence lengths
        self._create_qq_plot(
            label_lengths, 
            pred_lengths, 
            'QQ Plot: Sequence Lengths (Ground Truth vs Simulation)',
            os.path.join(self.output_dir, "qq_sequence_lengths.png"),
            log_scale=False
        )

    @staticmethod
    def build_count_process(event_times: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        return np.searchsorted(event_times, time_grid, side='right')

    @staticmethod
    def compute_cross_correlation(N: np.ndarray, h: int, r: int, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate E[(N(t+h)-N(t)) * (N(t+lag+r)-N(t+lag))] for various lags.
        Returns lags and the estimated cross-correlations.
        """
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
                values.append(0.0)  # padding if nothing is computable
        return lags, np.array(values)

    def plot_cross_correlation_moments(self, filename: str = "comparison_cross_correlation_moments.png"):
        """
        Plots and saves the cross-correlation of counting process increments.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")
        
        # Create time grids for count processes
        times_label = np.cumsum(self.label_time_deltas)
        times_pred = np.cumsum(self.simulated_time_deltas)
        
        # Skip if empty arrays
        if len(times_label) == 0 or len(times_pred) == 0:
            logger.warning("One or both time arrays are empty. Skipping cross-correlation plot.")
            return
        
        dt = 0.1
        T_max = max(np.max(times_label), np.max(times_pred)) + 1
        time_grid = np.arange(0, T_max, dt)

        N_label = self.build_count_process(times_label, time_grid)
        N_pred = self.build_count_process(times_pred, time_grid)

        h = r = int(2 / dt)  # corresponds to window size of 2 units
        max_lag = int(10 / dt)  # compute up to lag +/- 10 units

        lags, label_corr = self.compute_cross_correlation(N_label, h, r, max_lag)
        _, pred_corr = self.compute_cross_correlation(N_pred, h, r, max_lag)

        plt.figure(figsize=(10, 6))
        plt.plot(lags * dt, label_corr, label='Ground Truth', linewidth=2, color='royalblue')
        plt.plot(lags * dt, pred_corr, label='Simulation', linewidth=2, linestyle='--', color='crimson')
        plt.axvline(0, color='gray', linestyle=':', linewidth=1)
        
        plt.title('Cross-Correlation of Counting Process Increments', fontsize=14)
        plt.xlabel('Time lag (x - t)', fontsize=12)
        plt.ylabel(r'$E[\Delta N(t,h) \cdot \Delta N(x,r)]$', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        
        # Add annotation explaining the plot
        annotation_text = (f"Window sizes: h = r = {2} time units\n"
                          f"Shows temporal dependencies in event occurrence")
        plt.annotate(annotation_text, xy=(0.02, 0.02), xycoords='axes fraction',
                   va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Cross-correlation comparison plot saved to {filepath}")
    
    def run_evaluation(self) -> Dict[str, float]:
        """
        Run all evaluations and generate comparison plots between real and simulated data.
        """
        logger.info("Generating comparison plots for simulations...")
        
        try:
            # Generate all comparison plots
            self.plot_inter_event_time_distribution()
            self.plot_event_type_distribution()
            self.plot_sequence_length_distribution()
            self.plot_cross_correlation_moments()
            
            # Calculate summary metrics
            metrics = {
                "label_mean_time_delta": float(np.mean(self.label_time_deltas)),
                "simul_mean_time_delta": float(np.mean(self.simulated_time_deltas)),
                "label_median_time_delta": float(np.median(self.label_time_deltas)),
                "simul_median_time_delta": float(np.median(self.simulated_time_deltas)),
                "label_mean_seq_length": float(np.mean(self.label_sequence_lengths)),
                "simul_mean_seq_length": float(np.mean(self.simulated_sequence_lengths)),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"An error occurred during plot generation: {str(e)}", exc_info=True)
            return {}