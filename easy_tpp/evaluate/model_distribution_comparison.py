"""
Module for comparing distributions between real data and model simulations.
This module evaluates a model's ability to generate sequences with similar 
statistical properties to real data.
"""

from easy_tpp.utils import logger
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import DistribCompConfig, DataConfig

from typing import Dict
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class ModelDistributionComparator:
    """ 
    Class for comparing distributions of real data and model simulations.
    """
    
    def __init__(self,
                 evaluator_config: DistribCompConfig, 
                 model, 
                 dataset_size: int = 10**4):
        """
        Initialize the evaluator for comparing real data to model simulations.
        
        Args:
            evaluator_config: Configuration for the evaluator
            model: The trained model to evaluate
            dataset_size: Maximum number of events to use for distribution comparison
        """
        self.config = evaluator_config
        self.model = model
        
        # Initialize DataModule for the label data
        label_data_config_dict = evaluator_config.label_data_config
        data_specs_dict = evaluator_config.data_specs
        
        self.num_event_types = data_specs_dict['num_event_types']
        # Ensure data_specs is included in data configs
        label_data_config_dict['data_specs'] = data_specs_dict
        label_data_config_dict["data_loading_specs"] = evaluator_config.data_loading_specs
        
        label_data_config = DataConfig(**label_data_config_dict)
        
        self.label_split = evaluator_config.label_split
        
        self.label_loader_setup = TPPDataModule(label_data_config)
        self.label_loader_setup.setup(stage=evaluator_config.label_split)
        
        # Create output directory for saving plots
        self.output_dir = evaluator_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load label data
        label_source_dir = label_data_config.get_data_dir(split=self.label_split)
        label_data_format = label_data_config.data_format
        self.label_data = self.label_loader_setup.build_input(
            source_dir=label_source_dir, 
            split=self.label_split, 
            data_format=label_data_format
        )
        
        # Set the dataset size limit
        self.dataset_size = dataset_size
        
    def collect_real_data(self):
        """
        Collect real data events and statistics from the label dataset.
        """
        label_all_event_types = []
        label_all_time_deltas = []
        seq_lengths = []
        
        # Process events from each sequence
        i = 0
        for seq_idx in range(len(self.label_data['type_seqs'])):
            label_type_seq = self.label_data['type_seqs'][seq_idx]
            label_time_seq = self.label_data['time_delta_seqs'][seq_idx]
            
            seq_lengths.append(len(label_type_seq))
            
            for idx in range(len(label_type_seq)):
                label_all_event_types.append(label_type_seq[idx])
                label_all_time_deltas.append(label_time_seq[idx])
                i += 1
                if i >= self.dataset_size:
                    break
            
            if i >= self.dataset_size:
                break
        
        # Store collected data
        self.label_all_event_types = np.array(label_all_event_types[:self.dataset_size])
        self.label_all_time_deltas = np.array(label_all_time_deltas[:self.dataset_size])
        self.label_seq_lengths = np.array(seq_lengths)
        
        logger.info(f"Collected {len(self.label_all_event_types)} real events for comparison")
        
    def run_simulations(self, num_batches: int = 5):
        """
        Run model simulations and collect the generated events.
        
        Args:
            num_batches: Number of batches to use for simulation
        """
        device = next(self.model.parameters()).device
        simulated_all_event_types = []
        simulated_all_time_deltas = []
        simulated_seq_lengths = []
        
        # Get dataloader
        dataloader = self.label_loader_setup.get_dataloader(
            split=self.label_split, 
            shuffle=True,
            batch_size=self.model.simulation_batch_size
        )
        
        logger.info(f"Running model simulations with {num_batches} batches...")
        
        # Run simulations for specified number of batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                # Move batch to the model's device if needed
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Convert batch dictionary to tuple format expected by model.simulate
                batch_tuple = batch.values()
                
                # Run simulation
                time_seq, time_delta_seq, event_seq, simul_mask = self.model.simulate(
                    batch=batch_tuple
                )
                
                # Process simulation results
                batch_size = time_seq.size(0)
                for b in range(batch_size):
                    # Get valid events (where simul_mask is True)
                    mask = simul_mask[b]
                    valid_types = event_seq[b][mask].cpu().numpy()
                    valid_times = time_delta_seq[b][mask].cpu().numpy()
                    
                    # Store sequence length
                    seq_len = len(valid_types)
                    simulated_seq_lengths.append(seq_len)
                    
                    # Store individual events
                    for j in range(seq_len):
                        simulated_all_event_types.append(valid_types[j])
                        simulated_all_time_deltas.append(valid_times[j])
                        
                        if len(simulated_all_event_types) >= self.dataset_size:
                            break
                    
                    if len(simulated_all_event_types) >= self.dataset_size:
                        break
                
                if len(simulated_all_event_types) >= self.dataset_size:
                    break
                    
        # Store the collected simulated data
        self.simulated_all_event_types = np.array(simulated_all_event_types[:self.dataset_size])
        self.simulated_all_time_deltas = np.array(simulated_all_time_deltas[:self.dataset_size])
        self.simulated_seq_lengths = np.array(simulated_seq_lengths)
        
        logger.info(f"Collected {len(self.simulated_all_event_types)} simulated events for comparison")
        
    def plot_inter_event_time_distribution(self, filename: str = "comparison_inter_event_time_dist.png"):
        """
        Plots and saves the superimposed distribution of inter-event times between real and simulated data.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Main distribution plot
        plt.figure(figsize=(10, 6))
        
        # Use Seaborn's histplot or kdeplot for better aesthetics
        if self.label_all_time_deltas.size > 0 and self.simulated_all_time_deltas.size > 0:
            # Create histograms for both datasets
            label_hist, label_bin_edges = np.histogram(self.label_all_time_deltas, bins=50)
            pred_hist, pred_bin_edges = np.histogram(self.simulated_all_time_deltas, bins=50)
            
            # Get bin centers for plotting
            label_bin_centers = (label_bin_edges[:-1] + label_bin_edges[1:]) / 2
            pred_bin_centers = (pred_bin_edges[:-1] + pred_bin_edges[1:]) / 2
            
            # Filter out bins with frequency <= 2
            label_mask = label_hist > 2
            pred_mask = pred_hist > 2
            
            # Filter the histogram data and bin centers
            filtered_label_hist = label_hist[label_mask]
            filtered_label_bin_centers = label_bin_centers[label_mask]
            filtered_pred_hist = pred_hist[pred_mask]
            filtered_pred_bin_centers = pred_bin_centers[pred_mask]
            
            # Plot histograms - we'll use bar plots with filtered data instead of histplot
            plt.bar(filtered_label_bin_centers, filtered_label_hist, width=(label_bin_edges[1] - label_bin_edges[0]) * 0.8,
                  label=f'Real Data ({self.label_split})', alpha=0.6, color='royalblue')
            plt.bar(filtered_pred_bin_centers, filtered_pred_hist, width=(pred_bin_edges[1] - pred_bin_edges[0]) * 0.8,
                  label=f'Simulated Data', alpha=0.6, color='crimson')
            
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
                       label=f'Real data slope: {label_slope:.4f}')
            
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
                       label=f'Simulated slope: {pred_slope:.4f}')
            
            # Calculate statistics using vectorized operations
            label_mean = np.mean(self.label_all_time_deltas)
            label_median = np.median(self.label_all_time_deltas)
            label_std = np.std(self.label_all_time_deltas)
            
            pred_mean = np.mean(self.simulated_all_time_deltas)
            pred_median = np.median(self.simulated_all_time_deltas)
            pred_std = np.std(self.simulated_all_time_deltas)
            
            # Add statistics to the plot as annotations
            label_stats = (f"Real Data Stats:\n"
                         f"Mean: {label_mean:.4f}\n"
                         f"Median: {label_median:.4f}\n"
                         f"Std Dev: {label_std:.4f}")
            
            pred_stats = (f"Simulated Stats:\n"
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
            self.label_all_time_deltas, 
            self.simulated_all_time_deltas, 
            f'QQ Plot: Inter-Event Times (Real vs Simulated)',
            os.path.join(self.output_dir, "qq_inter_event_times.png"),
            log_scale=True
        )

    def _create_qq_plot(self, label_data: np.ndarray, pred_data: np.ndarray, title: str, save_path: str, log_scale: bool = False):
        """
        Creates and saves a QQ plot comparing real and simulated distributions.
        
        Args:
            label_data: Data points from the real dataset
            pred_data: Data points from the simulated dataset
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
        plt.xlabel(f'Real Data Quantiles ({self.label_split})', fontsize=12)
        plt.ylabel(f'Simulated Data Quantiles', fontsize=12)
        
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
        Plots and saves the superimposed distribution of event types between real and simulated data.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Get all possible event types - use numpy's unique function
        all_event_types = np.arange(self.num_event_types)
        
        # Count occurrences using numpy's bincount (faster than Counter for large arrays)
        # Ensure the arrays contain only integers
        label_types_int = self.label_all_event_types.astype(int)
        pred_types_int = self.simulated_all_event_types.astype(int)
        
        # Use bincount with minlength to ensure all event types are counted
        label_counts = np.bincount(label_types_int, minlength=self.num_event_types)
        pred_counts = np.bincount(pred_types_int, minlength=self.num_event_types)
        
        # Calculate normalized counts (probabilities) using numpy's division
        label_total = len(self.label_all_event_types)
        pred_total = len(self.simulated_all_event_types)
        
        label_probs = label_counts / label_total if label_total > 0 else np.zeros_like(label_counts)
        pred_probs = pred_counts / pred_total if pred_total > 0 else np.zeros_like(pred_counts)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Width of bars - slightly offset for better visibility
        width = 0.35
        x = np.arange(len(all_event_types))
        
        # Plot bars side by side
        plt.bar(x - width/2, label_probs, width, label=f'Real Data ({self.label_split})', 
              color='royalblue', alpha=0.7)
        plt.bar(x + width/2, pred_probs, width, label=f'Simulated Data', 
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
        label_stats = "Real Data Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {label_probs[i]:.3f}" 
                                                     for i in label_top3_indices])
        pred_stats = "Simulated Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {pred_probs[i]:.3f}" 
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
        Plots and saves the superimposed distribution of sequence lengths between real and simulated data.
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(10, 6))
        
        # Calculate appropriate bin width using vectorized operations
        # Use more robust calculation for bin width with defensive checks
        label_std = np.std(self.label_seq_lengths) if len(self.label_seq_lengths) > 1 else 1
        pred_std = np.std(self.simulated_seq_lengths) if len(self.simulated_seq_lengths) > 1 else 1
        
        # Take average of both standard deviations for bin width, minimum 1
        binwidth = max(1, int((label_std + pred_std) / 4))
        
        # Use Seaborn's histplot with optimized parameters
        sns.histplot(self.label_seq_lengths, label=f'Real Data ({self.label_split})', kde=True, 
                  stat='density', binwidth=binwidth, color='royalblue', alpha=0.6)
        sns.histplot(self.simulated_seq_lengths, label=f'Simulated Data', kde=True, 
                  stat='density', binwidth=binwidth, color='crimson', alpha=0.6)
        
        # Calculate statistics using vectorized operations
        label_mean = np.mean(self.label_seq_lengths)
        label_median = np.median(self.label_seq_lengths)
        label_std = np.std(self.label_seq_lengths)
        
        pred_mean = np.mean(self.simulated_seq_lengths)
        pred_median = np.median(self.simulated_seq_lengths)
        pred_std = np.std(self.simulated_seq_lengths)
        
        # Add statistics to the plot as annotations
        label_stats = (f"Real Data Stats:\n"
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
            self.label_seq_lengths, 
            self.simulated_seq_lengths, 
            f'QQ Plot: Sequence Lengths (Real vs Simulated)',
            os.path.join(self.output_dir, "qq_sequence_lengths.png"),
            log_scale=False
        )

    def run_evaluation(self, num_simulation_batches: int = 5) -> Dict[str, float]:
        """
        Run the full evaluation process:
        1. Collect real data
        2. Run model simulations
        3. Generate comparison plots
        
        Args:
            num_simulation_batches: Number of batches to use for simulation
            
        Returns:
            Dict[str, float]: Dictionary containing averaged metrics.
        """
        logger.info("Starting model evaluation with real vs. simulated data comparison...")
        
        try:
            # Collect real data from the label dataset
            logger.info("Collecting real data from the label dataset...")
            self.collect_real_data()
            
            # Run model simulations
            logger.info(f"Running model simulations with {num_simulation_batches} batches...")
            self.run_simulations(num_batches=num_simulation_batches)
            
            # Generate comparison plots
            logger.info("Generating comparison plots...")
            self.plot_inter_event_time_distribution()
            self.plot_event_type_distribution()
            self.plot_sequence_length_distribution()
            
            # Calculate KL divergence or other distribution metrics
            # Could be added here in future versions
            
            logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
            
        # Return an empty dictionary as a placeholder
        # Future versions could return actual metrics
        return {}

    @property
    def label_loader(self):
        self.label_loader_setup.setup(stage='test')
        return self.label_loader_setup.get_dataloader(split=self.label_split)