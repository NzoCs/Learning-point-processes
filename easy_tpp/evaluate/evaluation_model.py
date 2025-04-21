from easy_tpp.utils import logger
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import Config
from easy_tpp.evaluate.metrics_compute import MetricsCompute, EvaluationMode
from easy_tpp.preprocess.visualizer import Visualizer
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
        self.mode = EvaluationMode.SIMULATION  # Always use simulation mode
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {self.config.output_dir}")
        
        # Initialize data visualizer
        logger.info(f"Initializing Data Visualizer for {split} split...")
        self.data_visualizer = Visualizer(
            data_module=self.data_module, 
            split=self.split, 
            save_dir=os.path.join(self.config.output_dir, 'real_data_visuals')
        )
        
        # Storage for simulated data
        self.simulation_results = None
        
        # Get number of event types from the model
        self.num_event_types = self.model.num_event_types
        
        # Initialize metrics computer
        self.metrics_compute = MetricsCompute(
            mode=self.mode, 
            num_event_types=self.num_event_types
        )
        
        self.available_metrics = self.metrics_compute.get_available_metrics()
        logger.info(f"Available metrics: {self.available_metrics}")
        
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
    
    def plot_inter_event_time_distribution(self, sim_data: Dict[str, List]) -> None:
        """
        Plot and save the comparison of inter-event time distributions.
        
        Args:
            sim_data: Processed simulation data
        """
        real_times = self.data_visualizer.all_time_deltas
        sim_times = sim_data['time_deltas']
        
        if not real_times or not sim_times:
            logger.warning("Not enough data to plot inter-event time distribution")
            return
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(real_times, label=f'Real Data ({self.split})', 
                    fill=True, common_norm=False, log_scale=True, cut=0)
        sns.kdeplot(sim_times, label='Model Simulation', 
                    fill=True, common_norm=False, log_scale=True, cut=0)
        plt.title('Comparison of Inter-Event Time Distributions (Log Scale)')
        plt.xlabel('Time Since Last Event (Log Scale)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, "inter_event_time_dist.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Inter-event time distribution plot saved to {filepath}")
        
        # Add QQ plot
        self._create_qq_plot(
            real_times, 
            sim_times, 
            'QQ Plot: Inter-Event Times (Real vs Simulation)',
            os.path.join(self.config.output_dir, "qq_inter_event_times.png"),
            log_scale=True
        )
    
    def plot_event_type_distribution(self, sim_data: Dict[str, List]) -> None:
        """
        Plot and save the comparison of event type distributions.
        
        Args:
            sim_data: Processed simulation data
        """
        real_types = self.data_visualizer.all_event_types
        sim_types = sim_data['event_types']
        
        if not real_types or not sim_types:
            logger.warning("Not enough data to plot event type distribution")
            return
        
        df_real = pd.DataFrame({'type': real_types, 'source': f'Real Data ({self.split})'})
        df_sim = pd.DataFrame({'type': sim_types, 'source': 'Model Simulation'})
        df_combined = pd.concat([df_real, df_sim])
        
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(data=df_combined, x='type', hue='source', multiple='dodge', 
                          stat='proportion', common_norm=False, discrete=True, shrink=0.8)
        
        ax.set_xticks(range(self.num_event_types))
        ax.set_xticklabels(range(self.num_event_types))
        
        plt.title('Comparison of Event Type Distributions')
        plt.xlabel('Event Type')
        plt.ylabel('Proportion within Source')
        plt.legend(title='Data Source')
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, "event_type_dist.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Event type distribution plot saved to {filepath}")
        
        # Create QQ plot for event type proportions
        real_counts = pd.Series(real_types).value_counts(normalize=True).sort_index()
        sim_counts = pd.Series(sim_types).value_counts(normalize=True).sort_index()
        
        all_types = sorted(set(real_counts.index) | set(sim_counts.index))
        real_props = np.array([real_counts.get(t, 0) for t in all_types])
        sim_props = np.array([sim_counts.get(t, 0) for t in all_types])
        
        plt.figure(figsize=(8, 8))
        plt.plot(real_props, sim_props, 'o', markersize=6)
        
        max_prop = max(max(real_props), max(sim_props))
        ref_line = np.linspace(0, max_prop, 100)
        plt.plot(ref_line, ref_line, 'r--', alpha=0.7)
        
        for i, event_type in enumerate(all_types):
            plt.annotate(str(event_type), (real_props[i], sim_props[i]), 
                        textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.grid(True, alpha=0.3)
        plt.title('QQ Plot: Event Type Proportions (Real vs Simulation)')
        plt.xlabel(f'Real Data Proportions ({self.split})')
        plt.ylabel('Simulation Proportions')
        plt.axis('equal')
        plt.tight_layout()
        
        qq_filepath = os.path.join(self.config.output_dir, "qq_event_types.png")
        plt.savefig(qq_filepath)
        plt.close()
        logger.info(f"Event type QQ plot saved to {qq_filepath}")
    
    def plot_sequence_length_distribution(self, sim_data: Dict[str, List]) -> None:
        """
        Plot and save the comparison of sequence length distributions.
        
        Args:
            sim_data: Processed simulation data
        """
        real_lengths = [len(seq) for seq in self.data_visualizer.data["type_seqs"]]
        sim_lengths = sim_data['sequence_lengths']
        
        if not real_lengths or not sim_lengths:
            logger.warning("Not enough data to plot sequence length distribution")
            return
        
        plt.figure(figsize=(10, 6))
        binwidth_real = max(1, int(np.std(real_lengths)/2)) if real_lengths and np.std(real_lengths) > 0 else 1
        binwidth_sim = max(1, int(np.std(sim_lengths)/2)) if sim_lengths and np.std(sim_lengths) > 0 else 1
        binwidth = max(1, int((binwidth_real + binwidth_sim) / 2))
        
        sns.histplot(real_lengths, label=f'Real Data ({self.split})', kde=True, 
                     stat='density', common_norm=False, binwidth=binwidth)
        sns.histplot(sim_lengths, label='Model Simulation', kde=True, 
                     stat='density', common_norm=False, binwidth=binwidth)
        plt.title('Comparison of Sequence Length Distributions')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, "sequence_length_dist.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Sequence length distribution plot saved to {filepath}")
        
        # Add QQ plot
        self._create_qq_plot(
            real_lengths, 
            sim_lengths, 
            'QQ Plot: Sequence Lengths (Real vs Simulation)',
            os.path.join(self.config.output_dir, "qq_sequence_lengths.png"),
            log_scale=False
        )
    
    def _create_qq_plot(self, real_data: List, sim_data: List, title: str, save_path: str, log_scale: bool = False) -> None:
        """
        Create and save a QQ plot comparing real and simulated distributions.
        
        Args:
            real_data: Data points from the real dataset
            sim_data: Data points from the simulated dataset
            title: Title for the plot
            save_path: Path to save the plot
            log_scale: Whether to use log scale for the axes
        """
        # Filter out non-positive values if using log scale
        if log_scale:
            real_data = [x for x in real_data if x > 0]
            sim_data = [x for x in sim_data if x > 0]
        
        # Get quantiles for both datasets
        real_data = np.sort(real_data)
        sim_data = np.sort(sim_data)
        
        # Determine number of quantiles to use (minimum to avoid extrapolation)
        n_quantiles = min(len(real_data), len(sim_data))
        
        if n_quantiles < 10:
            logger.warning(f"Not enough data points for QQ plot: {n_quantiles} points")
            return
        
        quantiles = np.linspace(0, 1, n_quantiles)[1:-1]  # Exclude 0 and 1 quantiles
        
        real_quantiles = np.quantile(real_data, quantiles)
        sim_quantiles = np.quantile(sim_data, quantiles)
        
        # Create QQ plot
        plt.figure(figsize=(8, 8))
        
        # Plot the quantiles
        if log_scale:
            plt.loglog(real_quantiles, sim_quantiles, 'o', markersize=4)
            
            # Add reference line (y=x) in log-log space
            min_val = min(min(real_quantiles), min(sim_quantiles))
            max_val = max(max(real_quantiles), max(sim_quantiles))
            ref_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            plt.loglog(ref_line, ref_line, 'r--', alpha=0.7)
        else:
            plt.plot(real_quantiles, sim_quantiles, 'o', markersize=4)
            
            # Add reference line (y=x)
            min_val = min(min(real_quantiles), min(sim_quantiles))
            max_val = max(max(real_quantiles), max(sim_quantiles))
            ref_line = np.linspace(min_val, max_val, 100)
            plt.plot(ref_line, ref_line, 'r--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.xlabel(f'Real Data Quantiles ({self.split})')
        plt.ylabel('Simulation Quantiles')
        
        # Add annotation explaining interpretation
        plt.figtext(0.05, 0.01, 
                   "Points along reference line indicate similar distributions.\n"
                   "Deviations suggest distributional differences.",
                   fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"QQ plot saved to {save_path}")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics comparing real data with simulated data.
        
        Returns:
            Dictionary of computed metrics
        """
        # Make sure we have simulation results
        if self.simulation_results is None:
            logger.error("No simulation results available. Run simulations first.")
            return {}
        
        logger.info("Computing metrics between real and simulated data...")
        
        # Get data loaders
        real_loader = self.data_module.get_dataloader(split=self.split)
        
        # Prepare simulation data
        sim_times = self.simulation_results['time_seqs']
        sim_deltas = self.simulation_results['time_delta_seqs']
        sim_events = self.simulation_results['event_seqs']
        sim_masks = self.simulation_results['masks']
        
        # Initialize metrics collector
        all_metrics = []
        batch_count = 0
        
        # Go through each batch of real data and compare with corresponding simulation data
        for i, real_batch in enumerate(tqdm(real_loader, desc="Computing metrics", leave=False)):
            real_batch_data = list(real_batch.values())
            
            # Skip if we've gone through all simulation data
            if i * real_loader.batch_size >= sim_times.size(0):
                break
            
            # Extract corresponding simulation batch
            start_idx = i * real_loader.batch_size
            end_idx = min((i + 1) * real_loader.batch_size, sim_times.size(0))
            
            sim_batch_times = sim_times[start_idx:end_idx]
            sim_batch_deltas = sim_deltas[start_idx:end_idx]
            sim_batch_events = sim_events[start_idx:end_idx]
            sim_batch_masks = sim_masks[start_idx:end_idx]
            
            # Create simulation batch
            sim_batch = (sim_batch_times, sim_batch_deltas, sim_batch_events, sim_batch_masks, sim_batch_masks)
            
            try:
                # Compute metrics
                batch_metrics = self.metrics_compute.compute_all_metrics(
                    batch=real_batch_data,
                    pred=sim_batch
                )
                
                if batch_metrics:
                    all_metrics.append(batch_metrics)
                    batch_count += 1
            except Exception as e:
                logger.error(f"Error computing metrics for batch {i}: {e}", exc_info=True)
                continue
        
        # Average metrics across batches
        avg_metrics = {}
        if all_metrics:
            metric_keys = set(k for metrics in all_metrics for k in metrics.keys())
            for metric in metric_keys:
                values = [metrics[metric] for metrics in all_metrics 
                          if metric in metrics and metrics[metric] is not None and not np.isnan(metrics[metric])]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
                else:
                    avg_metrics[metric] = float('nan')
        
        logger.info(f"Metrics calculation complete: {batch_count} batches processed")
        
        return avg_metrics
    
    def save_metrics(self, metrics: Dict[str, float], filepath: str) -> None:
        """
        Save metrics dictionary to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filepath: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                serializable_metrics[k] = str(v)
            else:
                serializable_metrics[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def plot_metrics_table(self, metrics: Dict[str, float], title: str = "Evaluation Metrics") -> plt.Figure:
        """
        Plot a formatted table of evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics to display
            title: Title for the metrics table
            
        Returns:
            The matplotlib figure
        """
        time_metrics = {k: v for k, v in metrics.items() if 'time' in k.lower()}
        type_metrics = {k: v for k, v in metrics.items() if 'type' in k.lower() or 
                        any(term in k.lower() for term in ['accuracy', 'f1', 'recall', 'precision', 'entropy'])}
        sequence_metrics = {k: v for k, v in metrics.items() if 'sequence' in k.lower() or 
                          any(term in k.lower() for term in ['wasserstein', 'dtw', 'div'])}
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in time_metrics and k not in type_metrics and k not in sequence_metrics}
        
        data = []
        categories = []
        
        if time_metrics:
            data.append(list(time_metrics.items()))
            categories.append("Time Metrics")
        
        if type_metrics:
            data.append(list(type_metrics.items()))
            categories.append("Event Type Metrics")
        
        if sequence_metrics:
            data.append(list(sequence_metrics.items()))
            categories.append("Sequence Metrics")
        
        if other_metrics:
            data.append(list(other_metrics.items()))
            categories.append("Other Metrics")
        
        fig, ax = plt.subplots(figsize=(12, len(metrics) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for category_idx, category_metrics in enumerate(data):
            if category_idx > 0:
                table_data.append(["", ""])
            
            table_data.append([f"**{categories[category_idx]}**", ""])
            
            for metric_name, metric_value in category_metrics:
                formatted_name = " ".join(word.capitalize() for word in metric_name.replace('_', ' ').split())
                
                if isinstance(metric_value, float):
                    if np.isnan(metric_value):
                        formatted_value = "N/A"
                    elif metric_name.lower().endswith(('accuracy', 'f1score', 'recall', 'precision')):
                        formatted_value = f"{metric_value:.2f}%"
                    else:
                        formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                
                table_data.append([formatted_name, formatted_value])
        
        table = ax.table(cellText=table_data, 
                         colLabels=["Metric", "Value"], 
                         cellLoc='left', 
                         loc='center',
                         colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        
        category_rows = [i for i, row in enumerate(table_data) if row[0].startswith('**')]
        for row in category_rows:
            for j in range(2):
                cell = table._cells[(row + 1, j)]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
                if j == 0:
                    text = cell.get_text()
                    text.set_text(text.get_text().strip('*'))
        
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
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
        Run the full evaluation process.
        
        Returns:
            Dictionary containing evaluation metrics
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
        
        # Step 4: Generate plots if enabled
        if self.config.plot_figures:
            logger.info("Generating comparison plots...")
            try:
                self.plot_inter_event_time_distribution(sim_data)
                self.plot_event_type_distribution(sim_data)
                self.plot_sequence_length_distribution(sim_data)
            except Exception as e:
                logger.error(f"Error generating plots: {e}", exc_info=True)
        
        # Step 5: Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics = self.compute_metrics()
        
        # Step 6: Save metrics
        metrics_path = os.path.join(self.config.output_dir, 'evaluation_metrics.json')
        self.save_metrics(metrics, metrics_path)
        
        # Step 7: Generate metrics table
        if self.config.plot_figures and metrics:
            try:
                fig = self.plot_metrics_table(metrics, title="Model Evaluation Metrics")
                table_path = os.path.join(self.config.output_dir, 'metrics_table.png')
                fig.savefig(table_path, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Metrics table saved to {table_path}")
            except Exception as e:
                logger.error(f"Error generating metrics table: {e}", exc_info=True)
        
        logger.info(f"Evaluation complete. Results saved in {self.config.output_dir}")
        return metrics
