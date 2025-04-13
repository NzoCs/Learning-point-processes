from easy_tpp.utils import logger
from easy_tpp.preprocess import TPPDataModule
from easy_tpp.config_factory import EvaluationConfig, DataConfig, TokenizerConfig
from easy_tpp.evaluate.TPP_metrics_compute import TPPMetricsCompute, EvaluationMode


from typing import Dict, List, Union, Optional
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


class Evaluation :
    
    def __init__(self, evaluator_config: EvaluationConfig):
        """
        Initialize the evaluator with a specific mode.
        
        Args:
            evaluator_config: Configuration for the evaluator
        """
        
        # Initialize the dataloader
        label_data_config = evaluator_config.label_data_config
        pred_data_config = evaluator_config.pred_data_config
        data_specs = TokenizerConfig(**evaluator_config.data_specs)
        
        label_data_config['data_specs'] = data_specs
        pred_data_config['data_specs'] = data_specs
        
        label_data_config = DataConfig(**label_data_config)
        pred_data_config = DataConfig(**pred_data_config)
        self.label_split = evaluator_config.label_split
        self.pred_split = evaluator_config.pred_split
        
        self.num_event_types = data_specs.num_event_types
        self.label_loader_setup = TPPDataModule(label_data_config, batch_size = evaluator_config.batch_size)
        self.pred_loader_setup = TPPDataModule(pred_data_config, batch_size = evaluator_config.batch_size)
        
        
        # Set the mode
        mode = evaluator_config.mode
        try:
            self.mode = EvaluationMode(mode)
        except :
            raise ValueError(f'{mode} is not supported please initialize as simulation or prediction')
        
        logger.info(f"Evaluator initialized in {self.mode.value} mode")
        
        # Initialize available metrics based on mode
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """
        Initialize the available metrics based on the current mode.
        """
        if self.mode == EvaluationMode.PREDICTION:
            self.evaluator = TPPMetricsCompute(mode = EvaluationMode.PREDICTION, num_event_types = self.num_event_types)
            
        elif self.mode == EvaluationMode.SIMULATION:
            self.evaluator = TPPMetricsCompute(mode = EvaluationMode.SIMULATION, num_event_types = self.num_event_types)
        
        self.available_metrics = self.evaluator.get_available_metrics() if hasattr(self.evaluator, 'get_available_metrics') else []
        logger.info(f"Available metrics in {self.mode.value} mode: {self.available_metrics}")
    
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on all batches and return average metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing averaged metrics across all batches
        """
        all_batch_metrics = []
        total_batches = 0
        
        for label_batch, pred_batch in zip(self.label_loader, self.pred_loader):
            
            batch_metrics = self.evaluator.compute_all_metrics(batch=label_batch, pred=pred_batch)
            
            if batch_metrics:
                all_batch_metrics.append(batch_metrics)
                total_batches += 1
        
        # Calculate average metrics across all batches
        avg_metrics = {}
        if all_batch_metrics:
            for metric in all_batch_metrics[0].keys():
                # Filter out NaN values before computing mean
                values = [metrics[metric] for metrics in all_batch_metrics 
                          if metric in metrics and not np.isnan(metrics[metric])]
                if values:
                    avg_metrics[metric] = sum(values) / len(values)
                else:
                    avg_metrics[metric] = float('nan')
        
        logger.info(f"Evaluation complete: {len(all_batch_metrics)} batches processed")
        logger.info(f"Average metrics: {avg_metrics}")
        
        return avg_metrics
    
    def save_metrics(self, metrics: Dict[str, float], filepath: str) -> None:
        """
        Save metrics dictionary to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filepath: Path to save the JSON file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert any non-serializable values to strings
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                serializable_metrics[k] = str(v)
            else:
                serializable_metrics[k] = v
        
        # Write to JSON file
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
        # Group metrics by type
        time_metrics = {k: v for k, v in metrics.items() if 'time' in k.lower()}
        type_metrics = {k: v for k, v in metrics.items() if 'type' in k.lower() or 
                        any(term in k.lower() for term in ['accuracy', 'f1', 'recall', 'precision', 'entropy'])}
        sequence_metrics = {k: v for k, v in metrics.items() if 'sequence' in k.lower() or 
                          any(term in k.lower() for term in ['wasserstein', 'dtw', 'div'])}
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in time_metrics and k not in type_metrics and k not in sequence_metrics}
        
        # Create a DataFrame for the table
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
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(12, len(metrics) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Format table data
        table_data = []
        for category_idx, category_metrics in enumerate(data):
            if category_idx > 0:
                # Add separator between categories
                table_data.append(["", ""])
            
            # Add category header
            table_data.append([f"**{categories[category_idx]}**", ""])
            
            # Add metrics
            for metric_name, metric_value in category_metrics:
                # Format metric name to be more readable
                formatted_name = " ".join(word.capitalize() for word in metric_name.replace('_', ' ').split())
                
                # Format metric value
                if isinstance(metric_value, float):
                    if np.isnan(metric_value):
                        formatted_value = "N/A"
                    elif metric_name.lower().endswith(('accuracy', 'f1score', 'recall', 'precision')):
                        # Display as percentage with 2 decimal places
                        formatted_value = f"{metric_value:.2f}%"
                    else:
                        # Display with 4 decimal places
                        formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                
                table_data.append([formatted_name, formatted_value])
        
        # Create and display the table
        table = ax.table(cellText=table_data, 
                         colLabels=["Metric", "Value"], 
                         cellLoc='left', 
                         loc='center',
                         colWidths=[0.7, 0.3])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        # Style header
        for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        
        # Style category headers
        category_rows = [i for i, row in enumerate(table_data) if row[0].startswith('**')]
        for row in category_rows:
            for j in range(2):
                cell = table._cells[(row + 1, j)]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
                # Remove ** from text
                if j == 0:
                    text = cell.get_text()
                    text.set_text(text.get_text().strip('*'))
        
        # Add title
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @property
    def label_loader(self):
        return self.label_loader_setup.get_dataloader(split=self.label_split)
    
    @property
    def pred_loader(self):
        return self.pred_loader_setup.get_dataloader(split=self.pred_split)
    
    def set_mode(self, mode: Union[str, EvaluationMode]):
        """
        Change the evaluation mode.
        
        Args:
            mode: The new evaluation mode - either "point_process" or "sequence"
        """
        if isinstance(mode, str):
            try:
                self.mode = EvaluationMode(mode)
            except ValueError:
                logger.warning(f"Invalid mode: {mode}. Mode not changed.")
                return False
        else:
            self.mode = mode
            
        # Re-initialize the metrics for the new mode
        self._initialize_metrics()
        logger.info(f"Evaluator mode changed to {self.mode.value}")
        return True