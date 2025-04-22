import torch
import torch.nn.functional as F
import torchmetrics
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dataclasses import dataclass
from enum import Enum

from easy_tpp.utils import logger

# Define the evaluation modes
class EvaluationMode(Enum):
    SIMULATION = "simulation"
    PREDICTION = "prediction"

@dataclass
class MaskedValues:
    true_times: torch.Tensor
    true_types: torch.Tensor
    pred_times: torch.Tensor
    pred_types: torch.Tensor

class MetricsCompute:
    """
    Unified metrics computation class for Temporal Point Process models,
    compatible with EasyTPP's Gatech format data loaders.
    """
    
    def __init__(self, num_event_types: int, save_dir: str = None,
                 mode: EvaluationMode = EvaluationMode.PREDICTION):
        """
        Initialize metrics computation for TPP models.
        
        Args:
            num_event_types: Number of event types in the dataset
            mode: Prediction mode or Simulation mode
            save_dir: Directory for saving results
        """
        self.save_dir = save_dir
        self.num_event_types = num_event_types
        self.mode = mode
        
        if num_event_types <= 1:
            logger.info("Only one event type detected. Type prediction metrics will be skipped during evaluation.")
    
    def set_save_dir(self, save_dir: str):
        self.save_dir = save_dir

    def _create_attention_mask(self, non_pad_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = non_pad_mask.shape
        attention_mask = torch.ones((batch_size, seq_len, seq_len), device=non_pad_mask.device)
        attention_mask = attention_mask.tril(diagonal=0)
        mask_expanded = non_pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        attention_mask = attention_mask * mask_expanded
        return attention_mask

    def compute_all_metrics(self, batch, pred) -> Dict[str, float]:
        if self.mode == EvaluationMode.PREDICTION:
            return self._compute_prediction_metrics(batch, pred)
        else:
            return self._compute_simulation_metrics(batch, pred)
    
    def _compute_prediction_metrics(self, batch, pred) -> Dict[str, float]:
        metrics = {}
        # Extraction unique des valeurs masquées
        masked = self.get_masked_values(batch, pred)
        
        metrics['time_rmse'] = self.calculate_time_rmse(masked)
        metrics['time_mae'] = self.calculate_time_mae(masked)
        
        if self.num_event_types > 1:
            metrics['type_accuracy'] = self.calculate_type_accuracy(masked)
            metrics['macro_f1score'] = self.calculate_f1_score(masked, average='macro')
            metrics['micro_f1score'] = self.calculate_f1_score(masked, average='micro')
            metrics['recall'] = self.calculate_recall(masked)
            metrics['precision'] = self.calculate_precision(masked)
            metrics['cross_entropy'] = self.calculate_cross_entropy(masked)
            
        return metrics
    
    def _compute_simulation_metrics(self, batch, pred) -> Dict[str, float]:
        metrics = {}
        # Pour la simulation, on pourrait aussi encapsuler les valeurs via une structure spécifique
        # Ici, nous continuons à utiliser get_simulation_values séparément
        true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask = self.get_simulation_values(batch, pred)
        
        metrics['time_wasserstein'] = self.calculate_time_wasserstein(true_time_seqs, sim_time_seqs, sim_mask)
        metrics['sequence_dtw'] = self.calculate_sequence_dtw(true_time_seqs, true_type_seqs, sim_time_seqs, sim_type_seqs, sim_mask)
        metrics['sequence_kl_div'] = self.calculate_type_sequence_kl_div(true_type_seqs, sim_type_seqs, sim_mask)
        metrics['sequence_js_div'] = self.calculate_type_sequence_js_div(true_type_seqs, sim_type_seqs, sim_mask)
        
        return metrics

    def get_masked_values(self, batch, pred) -> MaskedValues:
        # Extraction selon le format EasyTPP
        if len(batch) >= 6:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask, _ = batch
        elif len(batch) >= 5:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask, attention_mask = batch
        else:
            raise ValueError("Batch values must contain at least 5 elements for prediction mode.")
        
        # Extraction des prédictions
        pred_time_delta_seqs, pred_type_seqs = pred[:2]
        
        mask = batch_non_pad_mask if batch_non_pad_mask is not None else torch.ones_like(true_type_seqs, dtype=torch.bool)
        
        true_times = true_time_delta_seqs[mask]
        true_types = true_type_seqs[mask]
        pred_times = pred_time_delta_seqs[mask]
        pred_types = pred_type_seqs[mask]
        
        return MaskedValues(true_times, true_types, pred_times, pred_types)

    # Les fonctions de calcul de métriques pour le mode PREDICTION reçoivent désormais l'objet MaskedValues
    def calculate_time_rmse(self, masked: MaskedValues) -> float:
        if masked.true_times.numel() == 0:
            return float('nan')
        # RMSE ne nécessite pas torchmetrics stateful, F.mse_loss gère les devices
        return torch.sqrt(F.mse_loss(masked.pred_times, masked.true_times)).item()
    
    def calculate_time_mae(self, masked: MaskedValues) -> float:
        if masked.true_times.numel() == 0:
            return float('nan')
        # MAE ne nécessite pas torchmetrics stateful, F.l1_loss gère les devices
        return F.l1_loss(masked.pred_times, masked.true_times).item()
    
    def calculate_type_accuracy(self, masked: MaskedValues) -> float:
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device # Get device from input tensor
        accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_event_types).to(device)
        accuracy_metric.update(masked.pred_types, masked.true_types)
        return accuracy_metric.compute().item() * 100
    
    def calculate_f1_score(self, masked: MaskedValues, average: str = 'macro') -> float:
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device # Get device from input tensor
        f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.num_event_types, average=average).to(device)
        f1_metric.update(masked.pred_types, masked.true_types)
        return f1_metric.compute().item() * 100
    
    def calculate_recall(self, masked: MaskedValues) -> float:
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device # Get device from input tensor
        recall_metric = torchmetrics.Recall(task="multiclass", num_classes=self.num_event_types, average='macro').to(device)
        recall_metric.update(masked.pred_types, masked.true_types)
        return recall_metric.compute().item() * 100
    
    def calculate_precision(self, masked: MaskedValues) -> float:
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        device = masked.true_types.device # Get device from input tensor
        precision_metric = torchmetrics.Precision(task="multiclass", num_classes=self.num_event_types, average='macro').to(device)
        precision_metric.update(masked.pred_types, masked.true_types)
        return precision_metric.compute().item() * 100
    
    def calculate_cross_entropy(self, masked: MaskedValues) -> float:
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float('nan')
        # Cross entropy ne nécessite pas torchmetrics stateful
        # Vérification sur le format des prédictions
        if masked.pred_types.dim() == 1:
            pred_logits = F.one_hot(masked.pred_types, num_classes=self.num_event_types).float()
        else:
            pred_logits = masked.pred_types
        loss = F.cross_entropy(pred_logits, masked.true_types)
        return loss.item()
    
    # Pour les métriques de simulation, on peut garder l’implémentation existante
    def get_simulation_values(self, batch, pred) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                            torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) >= 4:
            true_time_seqs, true_time_delta_seqs, true_type_seqs, batch_non_pad_mask = batch[:4]
        else:
            true_time_seqs, true_time_delta_seqs, true_type_seqs = batch[:3]
            batch_non_pad_mask = (true_type_seqs != self.num_event_types).float()
        
        sim_time_seqs, sim_time_delta_seqs, sim_type_seqs = pred[:3]
        sim_mask = pred[3] if len(pred) >= 4 else torch.ones_like(sim_type_seqs, dtype=torch.bool)
        return true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask
    
    def calculate_time_wasserstein(self, true_time_seqs: torch.Tensor, sim_time_seqs: torch.Tensor,
                                   sim_mask: torch.Tensor) -> float:
        true_times = true_time_seqs.cpu().numpy().flatten()
        sim_times = sim_time_seqs.cpu().numpy()[sim_mask.cpu().numpy()].flatten()
        if len(true_times) == 0 or len(sim_times) == 0:
            return float('nan')
        return float(wasserstein_distance(true_times, sim_times))
    
    def calculate_sequence_dtw(self, true_time_seqs: torch.Tensor, true_type_seqs: torch.Tensor,
                                 sim_time_seqs: torch.Tensor, sim_type_seqs: torch.Tensor,
                                 sim_mask: torch.Tensor) -> float:
        batch_size = true_time_seqs.shape[0]
        dtw_distances = []
        for i in range(batch_size):
            true_mask = (true_type_seqs[i] != self.num_event_types)
            true_seq_times = true_time_seqs[i][true_mask].cpu().numpy()
            true_seq_types = true_type_seqs[i][true_mask].cpu().numpy()
            true_seq = np.column_stack((true_seq_times, true_seq_types))
            
            sim_seq_mask = sim_mask[i]
            sim_seq_times = sim_time_seqs[i][sim_seq_mask].cpu().numpy()
            sim_seq_types = sim_type_seqs[i][sim_seq_mask].cpu().numpy()
            sim_seq = np.column_stack((sim_seq_times, sim_seq_types))
            
            if len(true_seq) == 0 or len(sim_seq) == 0:
                continue
                
            distance, _ = fastdtw(true_seq, sim_seq, dist=euclidean)
            dtw_distances.append(distance)
        
        if not dtw_distances:
            return float('nan')
        return float(np.mean(dtw_distances))
    
    def calculate_type_sequence_kl_div(self, true_type_seqs: torch.Tensor, sim_type_seqs: torch.Tensor,
                                       sim_mask: torch.Tensor) -> float:
        if self.num_event_types <= 1:
            return float('nan')
        true_type_dist = self._get_type_distribution(true_type_seqs)
        sim_type_dist = self._get_type_distribution(sim_type_seqs, sim_mask)
        kl_div = F.kl_div(torch.log(sim_type_dist + 1e-10), true_type_dist, reduction='batchmean')
        return kl_div.item()
    
    def calculate_type_sequence_js_div(self, true_type_seqs: torch.Tensor, sim_type_seqs: torch.Tensor,
                                       sim_mask: torch.Tensor) -> float:
        if self.num_event_types <= 1:
            return float('nan')
        true_type_dist = self._get_type_distribution(true_type_seqs).cpu().numpy()
        sim_type_dist = self._get_type_distribution(sim_type_seqs, sim_mask).cpu().numpy()
        js_div = jensenshannon(true_type_dist, sim_type_dist)
        return float(js_div)
    
    def _get_type_distribution(self, type_seqs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = (type_seqs != self.num_event_types)
        valid_types = type_seqs[mask]
        if len(valid_types) == 0:
            return torch.ones(self.num_event_types, device=type_seqs.device) / self.num_event_types
        type_counts = torch.bincount(valid_types, minlength=self.num_event_types)
        type_dist = type_counts.float() / type_counts.sum()
        return type_dist

    def get_available_metrics(self) -> List[str]:
        if self.mode == EvaluationMode.PREDICTION:
            metrics = ['time_rmse', 'time_mae']
            if self.num_event_types > 1:
                metrics.extend(['type_accuracy', 'macro_f1score', 'micro_f1score', 'recall', 'precision', 'cross_entropy'])
            return metrics
        elif self.mode == EvaluationMode.SIMULATION:
            return ['time_wasserstein', 'sequence_dtw', 'sequence_kl_div', 'sequence_js_div']
        else:
            return []
