from easy_tpp.utils import logger
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import EvaluationConfig, DataConfig, TokenizerConfig
from easy_tpp.evaluate.metrics_compute import MetricsCompute, EvaluationMode
from easy_tpp.preprocess.visualizer import Visualizer

from typing import Dict, List, Union, Optional, Tuple
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluation:
    
    def __init__(self, evaluator_config: EvaluationConfig):
        """
        Initialize the evaluator for simulation comparison.
        
        Args:
            evaluator_config: Configuration for the evaluator
        """
        self.config = evaluator_config
        self.mode = EvaluationMode.SIMULATION  # Hardcode mode to SIMULATION
        
        # Ensure output directory exists
        self.output_dir = evaluator_config.get('output_dir', 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {self.output_dir}")

        # Initialize DataModules first
        label_data_config_dict = evaluator_config.label_data_config
        pred_data_config_dict = evaluator_config.pred_data_config
        data_specs_dict = evaluator_config.data_specs
        
        # Ensure data_specs is included in data configs
        label_data_config_dict['data_specs'] = data_specs_dict
        pred_data_config_dict['data_specs'] = data_specs_dict
        
        label_data_config = DataConfig(**label_data_config_dict)
        pred_data_config = DataConfig(**pred_data_config_dict)
        
        self.label_split = evaluator_config.label_split
        self.pred_split = evaluator_config.pred_split
        
        self.label_loader_setup = TPPDataModule(label_data_config, batch_size=evaluator_config.batch_size)
        self.pred_loader_setup = TPPDataModule(pred_data_config, batch_size=evaluator_config.batch_size)

        # Initialize Visualizers using the DataModules
        logger.info("Initializing Label Visualizer...")
        self.label_visualizer = Visualizer(
            data_module=self.label_loader_setup, 
            split=self.label_split, 
            save_dir=os.path.join(self.output_dir, 'label_data_visuals')
        )
        logger.info("Initializing Prediction Visualizer...")
        self.pred_visualizer = Visualizer(
            data_module=self.pred_loader_setup, 
            split=self.pred_split, 
            save_dir=os.path.join(self.output_dir, 'pred_data_visuals')
        )

        # Get num_event_types from one of the visualizers
        self.num_event_types = self.label_visualizer.num_event_types

        logger.info(f"Evaluator initialized in {self.mode.value} mode")
        
        # Initialize available metrics based on mode
        self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initialize the available metrics for SIMULATION mode.
        """
        # Always initialize for SIMULATION mode
        self.evaluator = MetricsCompute(mode=EvaluationMode.SIMULATION, num_event_types=self.num_event_types)
        
        self.available_metrics = self.evaluator.get_available_metrics() if hasattr(self.evaluator, 'get_available_metrics') else []
        logger.info(f"Available metrics in {self.mode.value} mode: {self.available_metrics}")

    def plot_inter_event_time_distribution(self, label_times: List[float], pred_times: List[float], filename: str = "comparison_inter_event_time_dist.png"):
        """
        Plots and saves the superimposed distribution of inter-event times.
        Accepts raw time delta lists as input.
        """
        label_times_filtered = [t for t in label_times if t > 1e-9]
        pred_times_filtered = [t for t in pred_times if t > 1e-9]

        if not label_times_filtered or not pred_times_filtered:
            logger.warning("Not enough non-zero data to plot inter-event time distribution.")
            return

        plt.figure(figsize=(10, 6))
        sns.kdeplot(label_times_filtered, label=f'Label ({self.label_split})', fill=True, common_norm=False, log_scale=True, cut=0)
        sns.kdeplot(pred_times_filtered, label=f'Prediction ({self.pred_split})', fill=True, common_norm=False, log_scale=True, cut=0)
        plt.title('Comparison of Inter-Event Time Distributions (Log Scale)')
        plt.xlabel('Time Since Last Event (Log Scale)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Inter-event time distribution comparison plot saved to {filepath}")

    def plot_event_type_distribution(self, label_types: List[int], pred_types: List[int], filename: str = "comparison_event_type_dist.png"):
        """
        Plots and saves the superimposed distribution of event types.
        Accepts raw event type lists as input.
        """
        if not label_types or not pred_types:
            logger.warning("Not enough data to plot event type distribution.")
            return

        df_label = pd.DataFrame({'type': label_types, 'source': f'Label ({self.label_split})'})
        df_pred = pd.DataFrame({'type': pred_types, 'source': f'Prediction ({self.pred_split})'})
        df_combined = pd.concat([df_label, df_pred])

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
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Event type distribution comparison plot saved to {filepath}")

    def plot_sequence_length_distribution(self, label_lengths: List[int], pred_lengths: List[int], filename: str = "comparison_sequence_length_dist.png"):
        """
        Plots and saves the superimposed distribution of sequence lengths.
        Accepts raw sequence length lists as input.
        """
        if not label_lengths or not pred_lengths:
            logger.warning("Not enough data to plot sequence length distribution.")
            return

        plt.figure(figsize=(10, 6))
        binwidth_label = max(1, int(np.std(label_lengths)/2)) if label_lengths and np.std(label_lengths) > 0 else 1
        binwidth_pred = max(1, int(np.std(pred_lengths)/2)) if pred_lengths and np.std(pred_lengths) > 0 else 1
        binwidth = max(1, int((binwidth_label + binwidth_pred) / 2))
        
        sns.histplot(label_lengths, label=f'Label ({self.label_split})', kde=True, stat='density', common_norm=False, binwidth=binwidth)
        sns.histplot(pred_lengths, label=f'Prediction ({self.pred_split})', kde=True, stat='density', common_norm=False, binwidth=binwidth)
        plt.title('Comparison of Sequence Length Distributions')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Sequence length distribution comparison plot saved to {filepath}")

    def run_evaluation(self) -> Dict[str, float]:
        """
        Run the full evaluation process: calculate metrics and generate comparison plots.
        Uses Visualizer instances to get data for plotting.
        Assumes SIMULATION mode.
        
        Returns:
            Dict[str, float]: Dictionary containing averaged metrics.
        """
        logger.info(f"Starting comprehensive evaluation in {self.mode.value} mode...")

        logger.info("Calculating metrics...")
        all_batch_metrics = []
        total_batches = 0
        
        label_loader_iter = iter(self.label_loader)
        pred_loader_iter = iter(self.pred_loader)

        while True:
            try:
                label_batch = next(label_loader_iter)
                pred_batch = next(pred_loader_iter)
                
                if not label_batch or not pred_batch:
                    logger.warning("Skipping empty batch during metric calculation.")
                    continue

                batch_metrics = self.evaluator.compute_all_metrics(batch=label_batch, pred=pred_batch)
                
                if batch_metrics:
                    all_batch_metrics.append(batch_metrics)
                    total_batches += 1
            except StopIteration:
                break
            except Exception as e:
                 logger.error(f"Error processing batch for metrics: {e}", exc_info=True)
                 continue

        avg_metrics = {}
        if all_batch_metrics:
            metric_keys = set(k for metrics in all_batch_metrics for k in metrics.keys())
            for metric in metric_keys:
                values = [metrics[metric] for metrics in all_batch_metrics 
                          if metric in metrics and metrics[metric] is not None and not np.isnan(metrics[metric])]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
                else:
                    avg_metrics[metric] = float('nan')
        
        logger.info(f"Metrics calculation complete: {total_batches} batches processed")
        logger.info(f"Average metrics: {avg_metrics}")

        metrics_filepath = os.path.join(self.output_dir, 'evaluation_metrics.json')
        self.save_metrics(avg_metrics, metrics_filepath)

        logger.info("Generating comparison plots using data from Visualizers...")
        try:
            label_times = self.label_visualizer.all_time_deltas
            pred_times = self.pred_visualizer.all_time_deltas
            label_types = self.label_visualizer.all_event_types
            pred_types = self.pred_visualizer.all_event_types
            label_lengths = [len(seq) for seq in self.label_visualizer.data["type_seqs"]]
            pred_lengths = [len(seq) for seq in self.pred_visualizer.data["type_seqs"]]

            if label_times and pred_times and label_types and pred_types and label_lengths and pred_lengths:
                self.plot_inter_event_time_distribution(label_times, pred_times)
                self.plot_event_type_distribution(label_types, pred_types)
                self.plot_sequence_length_distribution(label_lengths, pred_lengths)
                
                try:
                    fig = self.plot_metrics_table(avg_metrics, title="Evaluation Metrics Summary")
                    table_filepath = os.path.join(self.output_dir, 'metrics_summary_table.png')
                    fig.savefig(table_filepath, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Metrics summary table plot saved to {table_filepath}")
                except Exception as e:
                    logger.error(f"Failed to generate or save metrics table plot: {e}", exc_info=True)
            else:
                logger.warning("Skipping comparison plot generation due to missing data in one or both visualizers.")
        except AttributeError as e:
            logger.error(f"Failed to access data from Visualizer instances. Ensure they initialized correctly. Error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during plot generation: {e}", exc_info=True)

        logger.info(f"Comprehensive evaluation finished. Results saved in {self.output_dir}")
        
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
    
    def plot_metrics_table(self, metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
        """
        Plot a formatted table of evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics to display
            title: Title for the metrics table
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
    
    @property
    def label_loader(self):
        if not hasattr(self.label_loader_setup, 'trainer'):
            self.label_loader_setup.setup(stage='test')
        return self.label_loader_setup.get_dataloader(split=self.label_split)
    
    @property
    def pred_loader(self):
        if not hasattr(self.pred_loader_setup, 'trainer'):
            self.pred_loader_setup.setup(stage='test')
        return self.pred_loader_setup.get_dataloader(split=self.pred_split)