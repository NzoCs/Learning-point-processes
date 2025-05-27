import torch
import torch.nn.functional as F
import torchmetrics
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
from scipy.stats import wasserstein_distance
from dataclasses import dataclass
from enum import Enum
from sklearn.metrics import roc_auc_score


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
        """
        Calculate simulation metrics using PyTorch batch operations.
        All computations are done on the same device as the input tensors.
        """
        metrics = {}
        true_time_seqs, true_type_seqs, true_time_delta_seqs, sim_time_seqs, sim_type_seqs, sim_mask = self.get_simulation_values(batch, pred)

        try:
            # Extract device from input tensors
            device = true_time_seqs.device if hasattr(true_time_seqs, 'device') else sim_time_seqs.device
            
            # Ensure all tensors are on the same device
            if not isinstance(true_time_seqs, torch.Tensor):
                true_time_seqs = torch.stack(true_time_seqs).to(device)
            if not isinstance(sim_time_seqs, torch.Tensor):
                sim_time_seqs = torch.stack(sim_time_seqs).to(device)
            if not isinstance(sim_mask, torch.Tensor):
                sim_mask = torch.stack(sim_mask).to(device)

            # Calculate Wasserstein 1D distance per sequence
            wasserstein_distances = self._batch_wasserstein_1d(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['wasserstein_1d'] = float(wasserstein_distances.mean().item())

            # Calculate MMD RBF with padding
            mmd_rbf = self._batch_mmd_rbf_padded(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['mmd_rbf_padded'] = float(mmd_rbf.item())

            # Calculate MMD with Wasserstein kernel
            mmd_wasserstein = self._batch_mmd_wasserstein(true_time_seqs, sim_time_seqs, sim_mask)
            metrics['mmd_wasserstein'] = float(mmd_wasserstein.item())

        except Exception as e:
            logger.error(f"Error computing simulation metrics: {e}")
            metrics = {
                'wasserstein_1d': float('nan'),
                'mmd_rbf_padded': float('nan'),
                'mmd_wasserstein': float('nan')
            }

        return metrics

    def _batch_wasserstein_1d(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calcule la distance de Wasserstein 1D pour chaque paire de séquences du batch.
        
        Args:
            true_seqs: [batch_size, seq_len] vraies séquences temporelles
            sim_seqs: [batch_size, seq_len] séquences simulées
            mask: [batch_size, seq_len] masque booléen
            
        Returns:
            torch.Tensor: [batch_size] distances de Wasserstein
        """
        batch_size, seq_len = true_seqs.shape
        # Appliquer mask et trier
        true_sorted = torch.sort(torch.where(mask, true_seqs, float('inf')), dim=1).values
        sim_sorted = torch.sort(torch.where(mask, sim_seqs, float('inf')), dim=1).values
        lengths = mask.sum(dim=1)
        dists = torch.zeros(batch_size, device=true_seqs.device)

        for i in range(batch_size):
            n = lengths[i].item()
            if n == 0:
                dists[i] = float('inf')
            else:
                x = true_sorted[i, :n].cpu().numpy()
                y = sim_sorted[i, :n].cpu().numpy()
                dists[i] = wasserstein_distance(x, y)
        return dists

    def _batch_mmd_rbf_padded(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        MMD^2 biaisée avec noyau RBF sur séquences padées.
        
        Args:
            true_seqs: [batch_size, seq_len] vraies séquences temporelles
            sim_seqs: [batch_size, seq_len] séquences simulées
            mask: [batch_size, seq_len] masque booléen
            sigma: paramètre de largeur de bande RBF
            
        Returns:
            torch.Tensor: valeur MMD au carré
        """
        batch_size, seq_len = true_seqs.shape
        # Calculer les longueurs et le max
        lengths = mask.sum(dim=1)
        max_len = int(lengths.max().item())
        if max_len == 0:
            return torch.tensor(float('nan'), device=true_seqs.device)

        # Préparer padding
        true_pad = torch.zeros(batch_size, max_len, device=true_seqs.device)
        sim_pad = torch.zeros(batch_size, max_len, device=sim_seqs.device)
        for i in range(batch_size):
            n = lengths[i].item()
            if n > 0:
                valid_true = true_seqs[i][mask[i]][:n]
                valid_sim = sim_seqs[i][mask[i]][:n]
                true_pad[i, :len(valid_true)] = valid_true
                sim_pad[i, :len(valid_sim)] = valid_sim

        # Calculer matrices de noyau
        def rbf(X, Y):
            # X: [batch, d], Y: [batch, d]
            XX = torch.cdist(X, X)**2
            YY = torch.cdist(Y, Y)**2
            XY = torch.cdist(X, Y)**2
            return torch.exp(-XX/(2*sigma**2)).mean(), \
                   torch.exp(-YY/(2*sigma**2)).mean(), \
                   torch.exp(-XY/(2*sigma**2)).mean()

        kxx, kyy, kxy = rbf(true_pad, sim_pad)
        return kxx + kyy - 2*kxy

    def _batch_mmd_wasserstein(self, true_seqs: torch.Tensor, sim_seqs: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        MMD^2 biaisée avec noyau RBF défini sur distances de Wasserstein.
        
        Args:
            true_seqs: [batch_size, seq_len] vraies séquences temporelles
            sim_seqs: [batch_size, seq_len] séquences simulées
            mask: [batch_size, seq_len] masque booléen
            sigma: paramètre de largeur de bande RBF
            
        Returns:
            torch.Tensor: valeur MMD au carré
        """
        batch_size = true_seqs.size(0)
        # Extraire sous-séquences valides
        true_list = []
        sim_list = []
        
        for i in range(batch_size):
            valid_mask = mask[i].bool()
            true_list.append(true_seqs[i][valid_mask].cpu().numpy())
            sim_list.append(sim_seqs[i][valid_mask].cpu().numpy())

        # Calculer matrices de distance de Wasserstein
        W_tt = np.zeros((batch_size, batch_size))
        W_ss = np.zeros((batch_size, batch_size))
        W_ts = np.zeros((batch_size, batch_size))
        
        for i in range(batch_size):
            for j in range(batch_size):
                W_tt[i,j] = wasserstein_distance(true_list[i], true_list[j])
                W_ss[i,j] = wasserstein_distance(sim_list[i], sim_list[j])
                W_ts[i,j] = wasserstein_distance(true_list[i], sim_list[j])

        # Convertir en torch et noyau
        T = torch.tensor(W_tt, device=true_seqs.device)
        S = torch.tensor(W_ss, device=true_seqs.device)
        X = torch.tensor(W_ts, device=true_seqs.device)

        K_tt = torch.exp(-T**2/(2*sigma**2)).mean()
        K_ss = torch.exp(-S**2/(2*sigma**2)).mean()
        K_ts = torch.exp(-X**2/(2*sigma**2)).mean()

        return K_tt + K_ss - 2*K_ts


    def classwise_auc_matrix(y_true: np.ndarray,
                            y_proba: np.ndarray,
                            class_names: Optional[List[str]] = None):
        """
        DataFrame pandas des AUC One-vs-Rest par classe.
        """
        import pandas as pd
        n_classes = y_proba.shape[1]
        aucs = []
        for i in range(n_classes):
            labels = (y_true == i).astype(int)
            aucs.append(roc_auc_score(labels, y_proba[:, i]))
        idx = class_names if class_names else [f'class_{i}' for i in range(n_classes)]
        return pd.DataFrame({'AUC': aucs}, index=idx)

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
    

    def get_available_metrics(self) -> List[str]:
        if self.mode == EvaluationMode.PREDICTION:
            metrics = ['time_rmse', 'time_mae']
            if self.num_event_types > 1:
                metrics.extend(['type_accuracy', 'macro_f1score', 'micro_f1score', 'recall', 'precision', 'cross_entropy'])
            return metrics
        elif self.mode == EvaluationMode.SIMULATION:
            return [
                'wasserstein_1d',
                'mmd_rbf_padded',
                'mmd_wasserstein'
            ]
        else:
            return []
