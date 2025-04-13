import torch
import torch.nn.functional as F
import torchmetrics
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from easy_tpp.utils import logger

from enum import Enum

# Define the evaluation modes
class EvaluationMode(Enum) :
    SIMULATION = "simulation"  
    PREDICTION = "prediction" 


class TPPMetricsCompute:
    """
    Unified metrics computation class for Temporal Point Process models,
    compatible with EasyTPP's Gatech format data loaders.
    """
    
    def __init__(
        self,
        num_event_types: int,
        save_dir : str = None,
        mode : EvaluationMode = EvaluationMode.PREDICTION
        ):
        """
        Initialize metrics computation for TPP models.
        
        Args:
            num_event_types: Number of event types in the dataset
            mode: Prediction mode - "one_step" for next event prediction or "simul" for full sequence simulation
            save_dir: Directory for saving results
        """
        
        self.save_dir = save_dir
        self.num_event_types = num_event_types
        self.mode = mode
        
        # Log at initialization if there's only one event type
        if num_event_types <= 1:
            logger.info("Only one event type detected. Type prediction metrics will be skipped during evaluation.")
    
    def set_save_dir(self, save_dir : str):  
        self.save_dir = save_dir
    
    def _create_attention_mask(self, non_pad_mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask where each event can only attend to its past events."""
        batch_size, seq_len = non_pad_mask.shape
        attention_mask = torch.ones((batch_size, seq_len, seq_len), device=non_pad_mask.device)
        
        # Lower triangular matrix (attend only to past)
        attention_mask = attention_mask.tril(diagonal=0)
        
        # Apply padding mask
        mask_expanded = non_pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = attention_mask * mask_expanded
        
        return attention_mask
    
    def compute_all_metrics(self, batch, pred) -> Dict[str, float]:
        """Compute all metrics based on the current mode."""
        if self.mode == EvaluationMode.PREDICTION:
            return self._compute_prediction_metrics(batch, pred)
        else:  # self.mode == EvaluationMode.SIMULATION
            return self._compute_simulation_metrics(batch, pred)
    
    def _compute_prediction_metrics(self, batch, pred) -> Dict[str, float]:
        """Compute metrics for one-step prediction using optimized batch operations."""
        metrics = {}
        
        # Calculate individual metrics using optimized batch operations
        metrics['time_rmse'] = self.calculate_time_rmse(batch, pred)
        metrics['time_mae'] = self.calculate_time_mae(batch, pred)
        
        # Only calculate type prediction metrics if we have more than one event type
        if self.num_event_types > 1:
            metrics['type_accuracy'] = self.calculate_type_accuracy(batch, pred)
            metrics['macro_f1score'] = self.calculate_f1_score(batch, pred, average='macro')
            metrics['micro_f1score'] = self.calculate_f1_score(batch, pred, average='micro')
            metrics['recall'] = self.calculate_recall(batch, pred)
            metrics['precision'] = self.calculate_precision(batch, pred)
            metrics['cross_entropy'] = self.calculate_cross_entropy(batch, pred)
        
        return metrics
    
    def _compute_simulation_metrics(self, batch, pred) -> Dict[str, float]:
        """Compute metrics for sequence simulation."""
        metrics = {}
        
        # Calculate individual metrics using optimized batch operations
        metrics['time_wasserstein'] = self.calculate_time_wasserstein(batch, pred)
        metrics['sequence_dtw'] = self.calculate_sequence_dtw(batch, pred)
        metrics['sequence_kl_div'] = self.calculate_type_sequence_kl_div(batch, pred)
        metrics['sequence_js_div'] = self.calculate_type_sequence_js_div(batch, pred)
        
        return metrics
    
    def get_masked_values(self, batch, pred) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract masked values for true and predicted times and types."""
        # Unpack the batch according to EasyTPP format
        if len(batch) >= 6:
            # Full EasyTPP format
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
        elif len(batch) >= 5:
            # Simplified format
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask = batch
            type_mask = None
        else:
            # Minimal format
            true_time_seqs, true_time_delta_seqs, true_type_seqs = batch[:3]
            batch_non_pad_mask = (true_type_seqs != self.num_event_types).float()
            attention_mask = self._create_attention_mask(batch_non_pad_mask)
            type_mask = None
        
        # Unpack predictions
        pred_time_delta_seqs, pred_type_seqs = pred[:2]
        
        # Create mask to extract non-padded elements
        if batch_non_pad_mask is not None:
            mask = batch_non_pad_mask.bool()
        else:
            # If no mask, assume all values are valid
            mask = torch.ones_like(true_type_seqs, dtype=torch.bool)
        
        # Apply mask to extract only valid elements
        true_times = true_time_delta_seqs[mask]
        true_types = true_type_seqs[mask]
        pred_times = pred_time_delta_seqs[mask]
        pred_types = pred_type_seqs[mask]
        
        return true_times, true_types, pred_times, pred_types
    
    def get_simulation_values(self, batch, pred) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract values for true and simulated times and types."""
        # Unpack the batch according to EasyTPP format
        if len(batch) >= 4:
            # Full simulation format
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask = batch[:4]
        else:
            # Minimal format
            true_time_seqs, true_time_delta_seqs, true_type_seqs = batch[:3]
            batch_non_pad_mask = (true_type_seqs != self.num_event_types).float()
        
        # Unpack simulations
        sim_time_seqs, sim_time_delta_seqs, sim_type_seqs = pred[:3]
        if len(pred) >= 4:
            sim_mask = pred[3]
        else:
            sim_mask = torch.ones_like(sim_type_seqs, dtype=torch.bool)
        
        return true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask
    
    def calculate_time_rmse(self, batch, pred) -> float:
        """Calculate Root Mean Squared Error for time predictions."""
        true_times, _, pred_times, _ = self.get_masked_values(batch, pred)
        
        if len(true_times) == 0:
            return float('nan')
            
        return torch.sqrt(F.mse_loss(pred_times, true_times)).item()
    
    def calculate_time_mae(self, batch, pred) -> float:
        """Calculate Mean Absolute Error for time predictions."""
        true_times, _, pred_times, _ = self.get_masked_values(batch, pred)
        
        if len(true_times) == 0:
            return float('nan')
            
        return F.l1_loss(pred_times, true_times).item()
    
    def calculate_type_accuracy(self, batch, pred) -> float:
        """Calculate event type prediction accuracy."""
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
            
        _, true_types, _, pred_types = self.get_masked_values(batch, pred)
        
        if len(true_types) == 0:
            return float('nan')
            
        accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_event_types)
        return accuracy_metric(pred_types, true_types).item() * 100
    
    def calculate_f1_score(self, batch, pred, average: str = 'macro') -> float:
        """Calculate F1 score for event type predictions.
        
        Args:
            average: Averaging method ('macro' or 'micro')
        """
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
            
        _, true_types, _, pred_types = self.get_masked_values(batch, pred)
        
        if len(true_types) == 0:
            return float('nan')
            
        f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.num_event_types, average=average)
        return f1_metric(pred_types, true_types).item() * 100
    
    def calculate_recall(self, batch, pred) -> float:
        """Calculate recall for event type predictions."""
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
            
        _, true_types, _, pred_types = self.get_masked_values(batch, pred)
        
        if len(true_types) == 0:
            return float('nan')
            
        recall_metric = torchmetrics.Recall(task="multiclass", num_classes=self.num_event_types, average='macro')
        return recall_metric(pred_types, true_types).item() * 100
    
    def calculate_precision(self, batch, pred) -> float:
        """Calculate precision for event type predictions."""
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
            
        _, true_types, _, pred_types = self.get_masked_values(batch, pred)
        
        if len(true_types) == 0:
            return float('nan')
            
        precision_metric = torchmetrics.Precision(task="multiclass", num_classes=self.num_event_types, average='macro')
        return precision_metric(pred_types, true_types).item() * 100
    
    def calculate_cross_entropy(self, batch, pred) -> float:
        """Calculate cross-entropy loss for event type predictions."""
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
            
        _, true_types, _, pred_types = self.get_masked_values(batch, pred)
        
        if len(true_types) == 0:
            return float('nan')
            
        # Convert predicted types to one-hot encoding (if needed)
        # For most cases, we need logits, not class predictions
        # If pred_types contains class indices, we need to convert to one-hot first
        if pred_types.dim() == 1:
            pred_logits = F.one_hot(pred_types, num_classes=self.num_event_types).float()
        else:
            pred_logits = pred_types
            
        # Calculate cross entropy loss
        loss = F.cross_entropy(pred_logits, true_types)
        return loss.item()
    
    def calculate_time_wasserstein(self, batch, pred) -> float:
        """Calculate Wasserstein distance between true and simulated time distributions."""
        true_time_seqs, _, true_time_delta_seqs, sim_time_seqs, _, sim_mask = self.get_simulation_values(batch, pred)
        
        # Convert to numpy for wasserstein calculation
        true_times = true_time_seqs.cpu().numpy().flatten()
        sim_times = sim_time_seqs.cpu().numpy()[sim_mask.cpu().numpy()].flatten()
        
        if len(true_times) == 0 or len(sim_times) == 0:
            return float('nan')
        
        # Calculate Wasserstein distance
        return float(wasserstein_distance(true_times, sim_times))
    
    def calculate_sequence_dtw(self, batch, pred) -> float:
        """Calculate Dynamic Time Warping distance between true and simulated sequences."""
        true_time_seqs, true_type_seqs, _, sim_time_seqs, sim_type_seqs, sim_mask = self.get_simulation_values(batch, pred)
        
        # Prepare sequence representations (time, type pairs)
        batch_size = true_time_seqs.shape[0]
        dtw_distances = []
        
        for i in range(batch_size):
            # Extract non-padded true sequence
            true_mask = (true_type_seqs[i] != self.num_event_types)
            true_seq_times = true_time_seqs[i][true_mask].cpu().numpy()
            true_seq_types = true_type_seqs[i][true_mask].cpu().numpy()
            true_seq = np.column_stack((true_seq_times, true_seq_types))
            
            # Extract simulated sequence
            sim_seq_mask = sim_mask[i]
            sim_seq_times = sim_time_seqs[i][sim_seq_mask].cpu().numpy()
            sim_seq_types = sim_type_seqs[i][sim_seq_mask].cpu().numpy()
            sim_seq = np.column_stack((sim_seq_times, sim_seq_types))
            
            if len(true_seq) == 0 or len(sim_seq) == 0:
                continue
                
            # Calculate DTW distance
            distance, _ = fastdtw(true_seq, sim_seq, dist = euclidean)
            dtw_distances.append(distance)
        
        if not dtw_distances:
            return float('nan')
            
        return float(np.mean(dtw_distances))
    
    def calculate_type_sequence_kl_div(self, batch, pred) -> float:
        """Calculate KL divergence between true and simulated type distributions."""
        _, true_type_seqs, _, _, sim_type_seqs, sim_mask = self.get_simulation_values(batch, pred)
        
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
        
        # Get type distributions
        true_type_dist = self._get_type_distribution(true_type_seqs)
        sim_type_dist = self._get_type_distribution(sim_type_seqs, sim_mask)
        
        # Calculate KL divergence
        kl_div = F.kl_div(
            torch.log(sim_type_dist + 1e-10),  # Add small epsilon to avoid log(0)
            true_type_dist,
            reduction='batchmean'
        )
        
        return kl_div.item()
    
    def calculate_type_sequence_js_div(self, batch, pred) -> float:
        """Calculate Jensen-Shannon divergence between true and simulated type distributions."""
        _, true_type_seqs, _, _, sim_type_seqs, sim_mask = self.get_simulation_values(batch, pred)
        
        # Skip if we only have one event type
        if self.num_event_types <= 1:
            return float('nan')
        
        # Get type distributions
        true_type_dist = self._get_type_distribution(true_type_seqs).cpu().numpy()
        sim_type_dist = self._get_type_distribution(sim_type_seqs, sim_mask).cpu().numpy()
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(true_type_dist, sim_type_dist)
        
        return float(js_div)
    
    
    def _get_type_distribution(self, type_seqs, mask=None) -> torch.Tensor:
        """Calculate the distribution of event types."""
        if mask is None:
            # For true sequences, consider non-padding values
            mask = (type_seqs != self.num_event_types)
        
        # Extract valid type values
        valid_types = type_seqs[mask]
        
        if len(valid_types) == 0:
            # Return uniform distribution if no valid types
            return torch.ones(self.num_event_types, device=type_seqs.device) / self.num_event_types
        
        # Count occurrences of each type
        type_counts = torch.bincount(valid_types, minlength=self.num_event_types)
        
        # Convert to probability distribution
        type_dist = type_counts.float() / type_counts.sum()
        
        return type_dist
    
    def get_available_metrics(self) -> List[str]:
        """Return a list of available metrics based on the current mode."""
        if self.mode == EvaluationMode.PREDICTION:
            metrics = [
                'time_rmse',
                'time_mae'
            ]
            
            # Only include type metrics if we have more than one event type
            if self.num_event_types > 1:
                metrics.extend([
                    'type_accuracy',
                    'macro_f1score',
                    'micro_f1score',
                    'recall',
                    'precision',
                    'cross_entropy'
                ])
            return metrics
        
        elif self.mode == EvaluationMode.SIMULATION:
            return [
                'time_wasserstein',
                'sequence_dtw',
                'sequence_kl_div',
                'sequence_js_div'
            ]
        else:
            return []