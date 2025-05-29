"""
Prediction metrics computation class.

This class is responsible solely for computing prediction-related metrics,
following the Single Responsibility Principle.
"""

import torch
import torch.nn.functional as F
import torchmetrics
from typing import Dict, List, Any
from .interfaces import MetricsComputerInterface, DataExtractorInterface
from .shared_types import MaskedValues
from easy_tpp.utils import logger


class PredictionDataExtractor(DataExtractorInterface):
    """Extracts prediction data from batch and predictions."""
    
    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
    
    def extract_values(self, batch: Any, pred: Any) -> MaskedValues:
        """Extract masked values for prediction metrics computation."""
        # Extraction according to EasyTPP format
        if len(batch) >= 6:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask, _ = batch
        elif len(batch) >= 5:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask = batch
        else:
            raise ValueError("Batch values must contain at least 5 elements for prediction mode.")
        
        # Extract predictions
        pred_time_delta_seqs, pred_type_seqs = pred[:2]
        
        mask = batch_non_pad_mask if batch_non_pad_mask is not None else torch.ones_like(true_type_seqs, dtype=torch.bool)
        
        true_times = true_time_delta_seqs[mask]
        true_types = true_type_seqs[mask]
        pred_times = pred_time_delta_seqs[mask]
        pred_types = pred_type_seqs[mask]
        
        return MaskedValues(true_times, true_types, pred_times, pred_types)


class PredictionMetricsComputer(MetricsComputerInterface):
    """
    Computes prediction-specific metrics.
    
    This class focuses solely on prediction metrics computation,
    adhering to the Single Responsibility Principle.
    """
    
    def __init__(self, num_event_types: int, data_extractor: DataExtractorInterface = None):
        """
        Initialize the prediction metrics computer.
        
        Args:
            num_event_types: Number of event types
            data_extractor: Custom data extractor (optional)
        """
        self.num_event_types = num_event_types
        self._data_extractor = data_extractor or PredictionDataExtractor(num_event_types)
    
    def compute_metrics(self, batch: Any, pred: Any) -> Dict[str, float]:
        """
        Compute all prediction metrics.
        
        Args:
            batch: Input batch data
            pred: Model predictions
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            metrics = {}
            masked = self._data_extractor.extract_values(batch, pred)
            
            # Always compute time-based metrics
            metrics['time_rmse'] = self._calculate_time_rmse(masked)
            metrics['time_mae'] = self._calculate_time_mae(masked)
            
            # Compute type-based metrics only for multi-class scenarios
            if self.num_event_types > 1:
                metrics['type_accuracy'] = self._calculate_type_accuracy(masked)
                metrics['macro_f1score'] = self._calculate_f1_score(masked, average='macro')
                metrics['micro_f1score'] = self._calculate_f1_score(masked, average='micro')
                metrics['recall'] = self._calculate_recall(masked)
                metrics['precision'] = self._calculate_precision(masked)
                metrics['cross_entropy'] = self._calculate_cross_entropy(masked)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing prediction metrics: {e}")
            return self._get_nan_metrics()
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available prediction metrics."""
        metrics = ['time_rmse', 'time_mae']
        if self.num_event_types > 1:
            metrics.extend([
                'type_accuracy', 'macro_f1score', 'micro_f1score', 
                'recall', 'precision', 'cross_entropy'
            ])
        return metrics
    
    def _calculate_time_rmse(self, masked: MaskedValues) -> float:
        """Calculate time-based RMSE."""
        if masked.true_times.numel() == 0:
            return float('nan')
        return torch.sqrt(F.mse_loss(masked.pred_times, masked.true_times)).item()
    
    def _calculate_time_mae(self, masked: MaskedValues) -> float:
        """Calculate time-based MAE."""
        if masked.true_times.numel() == 0:
            return float('nan')
        return F.l1_loss(masked.pred_times, masked.true_times).item()
    
    def _calculate_type_accuracy(self, masked: MaskedValues) -> float:
        """Calculate type classification accuracy."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device
        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=self.num_event_types
        ).to(device)
        accuracy_metric.update(masked.pred_types, masked.true_types)
        return accuracy_metric.compute().item() * 100
    
    def _calculate_f1_score(self, masked: MaskedValues, average: str = 'macro') -> float:
        """Calculate F1 score."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device
        f1_metric = torchmetrics.F1Score(
            task="multiclass", 
            num_classes=self.num_event_types, 
            average=average
        ).to(device)
        f1_metric.update(masked.pred_types, masked.true_types)
        return f1_metric.compute().item() * 100
    
    def _calculate_recall(self, masked: MaskedValues) -> float:
        """Calculate recall."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device
        recall_metric = torchmetrics.Recall(
            task="multiclass", 
            num_classes=self.num_event_types, 
            average='macro'
        ).to(device)
        recall_metric.update(masked.pred_types, masked.true_types)
        return recall_metric.compute().item() * 100
    
    def _calculate_precision(self, masked: MaskedValues) -> float:
        """Calculate precision."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device
        precision_metric = torchmetrics.Precision(
            task="multiclass", 
            num_classes=self.num_event_types, 
            average='macro'
        ).to(device)
        precision_metric.update(masked.pred_types, masked.true_types)
        return precision_metric.compute().item() * 100
    
    def _calculate_cross_entropy(self, masked: MaskedValues) -> float:
        """Calculate cross entropy loss."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        
        # Check format of predictions
        if masked.pred_types.dim() == 1:
            pred_logits = F.one_hot(masked.pred_types, num_classes=self.num_event_types).float()
        else:
            pred_logits = masked.pred_types
        
        loss = F.cross_entropy(pred_logits, masked.true_types)
        return loss.item()
    
    def _get_nan_metrics(self) -> Dict[str, float]:
        """Get a dictionary of NaN metrics for error cases."""
        metrics = {
            'time_rmse': float('nan'),
            'time_mae': float('nan')
        }
        if self.num_event_types > 1:
            metrics.update({
                'type_accuracy': float('nan'),
                'macro_f1score': float('nan'),
                'micro_f1score': float('nan'),
                'recall': float('nan'),
                'precision': float('nan'),
                'cross_entropy': float('nan')
            })
        return metrics
