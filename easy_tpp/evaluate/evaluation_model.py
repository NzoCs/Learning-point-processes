from easy_tpp.utils import logger
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import Config
from easy_tpp.models.basemodel import BaseModel

from typing import Dict, List, Union, Optional, Tuple, Any
import torch
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

class ModelEvaluationConfig(Config):
    """Configuration class for model evaluation.
    
    Attributes:
        output_dir (str): Directory to save evaluation results
        batch_size (int): Batch size for data loading
        num_simulations (int): Number of simulations to run
        plot_figures (bool): Whether to generate plots
    """
    
    def __init__(self, 
                 output_dir: str = 'model_evaluation_results',
                 batch_size: int = 32,
                 num_simulations: int = 10,
                 plot_figures: bool = True) -> None:
        """Initialize ModelEvaluationConfig with type-safe parameters."""
        self.output_dir = output_dir
        self.batch_size = int(batch_size)
        self.num_simulations = int(num_simulations)
        self.plot_figures = bool(plot_figures)
        
    def get_yaml_config(self) -> dict:
        """Return the config in dict (yaml compatible) format."""
        return {
            'output_dir': self.output_dir,
            'batch_size': self.batch_size,
            'num_simulations': self.num_simulations,
            'plot_figures': self.plot_figures
        }
        
    @staticmethod
    def parse_from_yaml_config(yaml_config) -> 'ModelEvaluationConfig':
        """Parse from yaml to generate the config object."""
        return ModelEvaluationConfig(**yaml_config) if yaml_config is not None else None


class ModelEvaluation:
    """Evaluation class for comparing real data with model simulations."""
    
    def __init__(self, 
                 model: BaseModel, 
                 data_module: TPPDataModule, 
                 config: ModelEvaluationConfig = None,
                 split: str = 'test'):
        """
        Initialize the evaluator for comparing real data with model simulations.
        
        Args:
            model: The trained model to evaluate
            data_module: Data module containing the real data
            config: Configuration for the evaluator
            split: Data split to use for evaluation ('train', 'val', or 'test')
        """
        self.model = model
        self.data_module = data_module
        self.config = config or ModelEvaluationConfig()
        self.split = split

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {self.config.output_dir}")
        
        # Storage for simulated data
        self.simulation_results = None
        
        # Get number of event types from the model
        self.num_event_types = self.model.num_event_types
        
    def run_simulations(self) -> Dict[str, torch.Tensor]:
        """
        Run simulations using the model.
        
        Returns:
            Dict containing simulated data tensors
        """
        logger.info(f"Running {self.config.num_simulations} simulations...")
        
        # Initialize collectors for simulated data
        all_time_seqs = []
        all_time_delta_seqs = []
        all_event_seqs = []
        all_masks = []
        
        # Prepare model for simulation
        self.model.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(self.config.num_simulations), desc="Simulating sequences"):
                # Run simulation using the model's simulate method
                try:
                    time_seq, time_delta_seq, event_seq, mask = self.model.simulate()
                    
                    # Move to CPU for storage
                    all_time_seqs.append(time_seq.cpu())
                    all_time_delta_seqs.append(time_delta_seq.cpu())
                    all_event_seqs.append(event_seq.cpu())
                    all_masks.append(mask.cpu())
                except Exception as e:
                    logger.error(f"Error during simulation: {e}", exc_info=True)
        
        # Concatenate results
        if all_time_seqs:
            self.simulation_results = {
                'time_seqs': torch.cat(all_time_seqs, dim=0),
                'time_delta_seqs': torch.cat(all_time_delta_seqs, dim=0),
                'event_seqs': torch.cat(all_event_seqs, dim=0),
                'masks': torch.cat(all_masks, dim=0)
            }
            logger.info(f"Generated {self.simulation_results['time_seqs'].size(0)} simulated sequences")
        else:
            logger.error("Failed to generate any simulations")
            self.simulation_results = None
            
        return self.simulation_results
    
    def _prepare_simulation_data(self) -> Dict[str, List]:
        """
        Prepare simulated data for visualization and metrics computation.
        
        Returns:
            Dict containing processed simulation data
        """
        if self.simulation_results is None:
            logger.error("No simulation results available. Run simulations first.")
            return None
        
        # Extract tensors
        time_seqs = self.simulation_results['time_seqs']
        time_delta_seqs = self.simulation_results['time_delta_seqs']
        event_seqs = self.simulation_results['event_seqs']
        masks = self.simulation_results['masks']
        
        # Convert to lists and apply masks
        all_time_deltas = []
        all_event_types = []
        all_sequence_lengths = []
        
        batch_size, max_seq_len = time_seqs.size()
        
        for b in range(batch_size):
            # Get valid indices from mask
            valid_indices = masks[b].bool()
            
            # Extract valid elements
            valid_time_deltas = time_delta_seqs[b][valid_indices].tolist()
            valid_event_types = event_seqs[b][valid_indices].tolist()
            
            sequence_length = valid_indices.sum().item()
            
            all_time_deltas.extend(valid_time_deltas)
            all_event_types.extend(valid_event_types)
            all_sequence_lengths.append(sequence_length)
        
        return {
            'time_deltas': all_time_deltas,
            'event_types': all_event_types,
            'sequence_lengths': all_sequence_lengths
        }
    
    def plot_inter_event_time_distribution(self, sim_data: Dict[str, List], filename: str = "comparison_inter_event_time_dist.png"):
        """
        Plots and saves the superimposed distribution of inter-event times.
        
        Args:
            sim_data: Dictionary containing the processed simulation data
            filename: Name of the output file
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Extract data
        real_times = np.asarray(self.data_visualizer.all_time_deltas)
        sim_times = np.asarray(sim_data['time_deltas'])
        
        # Main distribution plot
        plt.figure(figsize=(10, 6))
        
        # Use histplot with KDE for better visualization
        if real_times.size > 0 and sim_times.size > 0:
            sns.histplot(real_times, label=f'Real Data ({self.split})', kde=True, 
                      alpha=0.6, log_scale=True, color='royalblue')
            sns.histplot(sim_times, label='Model Simulation', kde=True, 
                      alpha=0.6, log_scale=True, color='crimson')
            
            # Calculate statistics
            real_mean = np.mean(real_times)
            real_median = np.median(real_times)
            real_std = np.std(real_times)
            
            sim_mean = np.mean(sim_times)
            sim_median = np.median(sim_times)
            sim_std = np.std(sim_times)
            
            # Add statistics to the plot as annotations
            real_stats = (f"Real Data Stats:\n"
                         f"Mean: {real_mean:.4f}\n"
                         f"Median: {real_median:.4f}\n"
                         f"Std Dev: {real_std:.4f}")
            
            sim_stats = (f"Simulation Stats:\n"
                        f"Mean: {sim_mean:.4f}\n"
                        f"Median: {sim_median:.4f}\n"
                        f"Std Dev: {sim_std:.4f}")
            
            # Position text for real data stats at top left
            plt.annotate(real_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                      va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            # Position text for simulation stats at top right
            plt.annotate(sim_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                      va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.title('Comparison of Inter-Event Time Distributions (Log Scale)')
        plt.xlabel('Time Since Last Event (Log Scale)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Inter-event time distribution comparison plot saved to {filepath}")
        
        # Add QQ plot for inter-event times
        self._create_qq_plot(
            real_times, 
            sim_times, 
            f'QQ Plot: Inter-Event Times (Real vs Simulation)',
            os.path.join(self.config.output_dir, "qq_inter_event_times.png"),
            log_scale=True
        )
    
    def plot_event_type_distribution(self, sim_data: Dict[str, List], filename: str = "comparison_event_type_dist.png"):
        """
        Plots and saves the superimposed distribution of event types.
        
        Args:
            sim_data: Dictionary containing the processed simulation data
            filename: Name of the output file
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Extract data
        real_types = np.asarray(self.data_visualizer.all_event_types)
        sim_types = np.asarray(sim_data['event_types'])
        
        # Get all possible event types
        all_event_types = np.arange(self.num_event_types)
        
        # Count occurrences using numpy's bincount (faster than Counter for large arrays)
        real_types_int = real_types.astype(int)
        sim_types_int = sim_types.astype(int)
        
        # Use bincount with minlength to ensure all event types are counted
        real_counts = np.bincount(real_types_int, minlength=self.num_event_types)
        sim_counts = np.bincount(sim_types_int, minlength=self.num_event_types)
        
        # Calculate normalized counts (probabilities)
        real_total = len(real_types)
        sim_total = len(sim_types)
        
        real_probs = real_counts / real_total if real_total > 0 else np.zeros_like(real_counts)
        sim_probs = sim_counts / sim_total if sim_total > 0 else np.zeros_like(sim_counts)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Width of bars - slightly offset for better visibility
        width = 0.35
        x = np.arange(len(all_event_types))
        
        # Plot bars side by side
        plt.bar(x - width/2, real_probs, width, label=f'Real Data ({self.split})', 
              color='royalblue', alpha=0.7)
        plt.bar(x + width/2, sim_probs, width, label='Model Simulation', 
              color='crimson', alpha=0.7)
        
        plt.title('Comparison of Event Type Distributions', fontsize=14)
        plt.xlabel('Event Type', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(x, all_event_types)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        
        # Add statistics for top event types as annotations
        # Use numpy's argsort to get indices of top event types
        real_top3_indices = np.argsort(real_probs)[-3:][::-1]  # Get top 3 in descending order
        sim_top3_indices = np.argsort(sim_probs)[-3:][::-1]
        
        # Create formatted statistics strings
        real_stats = "Real Data Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {real_probs[i]:.3f}" 
                                                     for i in real_top3_indices])
        sim_stats = "Simulation Top Types:\n" + "\n".join([f"Type {all_event_types[i]}: {sim_probs[i]:.3f}" 
                                                         for i in sim_top3_indices])
        
        # Position text on left and right sides
        plt.annotate(real_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.annotate(sim_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Event type distribution comparison plot saved to {filepath}")
    
    def plot_sequence_length_distribution(self, sim_data: Dict[str, List], filename: str = "comparison_sequence_length_dist.png"):
        """
        Plots and saves the superimposed distribution of sequence lengths.
        
        Args:
            sim_data: Dictionary containing the processed simulation data
            filename: Name of the output file
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")

        # Extract data
        real_lengths = np.array([len(seq) for seq in self.data_module.get_data(self.split)["type_seqs"]])
        sim_lengths = np.array(sim_data['sequence_lengths'])
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Calculate appropriate bin width using vectorized operations
        real_std = np.std(real_lengths) if len(real_lengths) > 1 else 1
        sim_std = np.std(sim_lengths) if len(sim_lengths) > 1 else 1
        
        # Take average of both standard deviations for bin width, minimum 1
        binwidth = max(1, int((real_std + sim_std) / 4))
        
        # Use Seaborn's histplot with optimized parameters
        sns.histplot(real_lengths, label=f'Real Data ({self.split})', kde=True, 
                  stat='density', binwidth=binwidth, color='royalblue', alpha=0.6)
        sns.histplot(sim_lengths, label='Model Simulation', kde=True, 
                  stat='density', binwidth=binwidth, color='crimson', alpha=0.6)
        
        # Calculate statistics using vectorized operations
        real_mean = np.mean(real_lengths)
        real_median = np.median(real_lengths)
        real_std = np.std(real_lengths)
        
        sim_mean = np.mean(sim_lengths)
        sim_median = np.median(sim_lengths)
        sim_std = np.std(sim_lengths)
        
        # Add statistics to the plot as annotations
        real_stats = (f"Real Data Stats:\n"
                     f"Mean: {real_mean:.2f}\n"
                     f"Median: {real_median:.2f}\n"
                     f"Std Dev: {real_std:.2f}")
        
        sim_stats = (f"Simulation Stats:\n"
                    f"Mean: {sim_mean:.2f}\n"
                    f"Median: {sim_median:.2f}\n"
                    f"Std Dev: {sim_std:.2f}")
        
        # Position text for real data stats at top left
        plt.annotate(real_stats, xy=(0.05, 0.95), xycoords='axes fraction',
                   va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        # Position text for simulation stats at top right
        plt.annotate(sim_stats, xy=(0.95, 0.95), xycoords='axes fraction',
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                   fontsize=10)
        
        plt.title('Comparison of Sequence Length Distributions', fontsize=14)
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Data Source', frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Sequence length distribution comparison plot saved to {filepath}")
        
        # Add QQ plot for sequence lengths
        self._create_qq_plot(
            real_lengths, 
            sim_lengths, 
            f'QQ Plot: Sequence Lengths (Real vs Simulation)',
            os.path.join(self.config.output_dir, "qq_sequence_lengths.png"),
            log_scale=False
        )
    
    def _create_qq_plot(self, real_data: Union[List, np.ndarray], sim_data: Union[List, np.ndarray], title: str, save_path: str, log_scale: bool = False) -> None:
        """
        Creates and saves a QQ plot comparing real and simulated distributions.
        
        Args:
            real_data: Data points from the real dataset
            sim_data: Data points from the simulated dataset
            title: Title for the plot
            save_path: Path to save the plot
            log_scale: Whether to use log scale for the axes
        """
        # Set the Seaborn style and context
        sns.set_theme(style="whitegrid")
        
        # Convert inputs to numpy arrays if they aren't already
        real_data = np.asarray(real_data)
        sim_data = np.asarray(sim_data)
        
        # Filter out non-positive values if using log scale - use vectorized operations
        if log_scale:
            real_data = real_data[real_data > 0]
            sim_data = sim_data[sim_data > 0]
        
        # Skip if empty arrays
        if len(real_data) == 0 or len(sim_data) == 0:
            logger.warning(f"Empty arrays after filtering: real_data={len(real_data)}, sim_data={len(sim_data)}")
            return
            
        # Get quantiles using vectorized operations
        # Sort the arrays in-place for better memory efficiency
        real_data.sort()
        sim_data.sort()
        
        # Determine number of quantiles to use (minimum to avoid extrapolation)
        n_quantiles = min(len(real_data), len(sim_data))
        
        if n_quantiles < 10:
            logger.warning(f"Not enough data points for QQ plot: {n_quantiles} points")
            return
        
        # Create evenly spaced quantiles, avoiding 0 and 1 to prevent infinity issues with some distributions
        quantiles = np.linspace(0.01, 0.99, min(100, n_quantiles))
        
        # Compute quantiles efficiently using vectorized numpy functions
        real_quantiles = np.quantile(real_data, quantiles)
        sim_quantiles = np.quantile(sim_data, quantiles)
        
        # Create QQ plot
        plt.figure(figsize=(8, 8))
        
        # Plot the quantiles
        if log_scale:
            plt.loglog(real_quantiles, sim_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x) in log-log space
            min_val = max(min(real_quantiles.min(), sim_quantiles.min()), 1e-10)  # Avoid log(0)
            max_val = max(real_quantiles.max(), sim_quantiles.max())
            ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            plt.loglog(ref_line, ref_line, 'r--', alpha=0.7)
        else:
            plt.plot(real_quantiles, sim_quantiles, 'o', markersize=4, color='royalblue', alpha=0.7)
            
            # Add reference line (y=x)
            min_val = min(real_quantiles.min(), sim_quantiles.min())
            max_val = max(real_quantiles.max(), sim_quantiles.max())
            ref_line = np.linspace(min_val, max_val, 100)
            plt.plot(ref_line, ref_line, 'r--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.title(title, fontsize=14)
        plt.xlabel(f'Real Data Quantiles ({self.split})', fontsize=12)
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
    
    def save_model_metadata(self) -> None:
        """Save model metadata for reference."""
        try:
            metadata = self.model.get_model_metadata()
            metadata_path = os.path.join(self.config.output_dir, 'model_metadata.json')
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Model metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}", exc_info=True)
    
    def run_evaluation(self) -> Dict[str, float]:
        """
        Run the full evaluation process: generate simulations and compare with real data through plots.
        
        Returns:
            Dictionary containing evaluation metrics (empty in this case)
        """
        logger.info("Starting model evaluation...")
        
        # Step 1: Run simulations
        self.run_simulations()
        if self.simulation_results is None:
            logger.error("Failed to generate simulations. Evaluation aborted.")
            return {}
        
        # Step 2: Save model metadata
        self.save_model_metadata()
        
        # Step 3: Process simulation data
        sim_data = self._prepare_simulation_data()
        if not sim_data:
            logger.error("Failed to process simulation data. Evaluation aborted.")
            return {}
        
        # Step 4: Generate comparison plots
        logger.info("Generating comparison plots between real and simulated data...")
        try:
            # Extract the data for plotting
            real_times = self.data_visualizer.all_time_deltas
            sim_times = sim_data['time_deltas']
            
            real_types = self.data_visualizer.all_event_types
            sim_types = sim_data['event_types']
            
            real_lengths = [len(seq) for seq in self.data_module.get_data(self.split)["type_seqs"]]
            sim_lengths = sim_data['sequence_lengths']

            # Generate plots
            self.plot_inter_event_time_distribution(sim_data)
            self.plot_event_type_distribution(sim_data)
            self.plot_sequence_length_distribution(sim_data)
        except Exception as e:
            logger.error(f"Error generating plots: {e}", exc_info=True)
        
        logger.info(f"Evaluation complete. Results saved in {self.config.output_dir}")
        return {}
