"""
Prediction metrics computation class.
"""

from typing import Any, Dict, List, Optional, Union, Literal

import torch
import torch.nn.functional as F
import torchmetrics

from new_ltpp.shared_types import Batch, OneStepPrediction
from new_ltpp.utils import logger

from .metrics_interfaces import (
    DataExtractorInterface,
    MetricsComputerInterface,
    TimeExtractorInterface,
    TypeExtractorInterface,
)
from .shared_types import MaskedValues, PredictionMetrics, TimeValues, TypeValues


class TimeDataExtractor(TimeExtractorInterface):
    """Extracts time-related data from batch and predictions."""

    def extract_time_values(self, batch: Batch, pred: OneStepPrediction) -> TimeValues:
        """
        Extract masked time values for prediction metrics computation.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with dtime_predict
        """
        # Extract from Batch dataclass
        true_time_delta_seqs = batch.time_delta_seqs
        true_type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        pred_time_delta_seqs = pred.dtime_predict

        mask = (
            batch_non_pad_mask
            if batch_non_pad_mask is not None
            else torch.ones_like(true_type_seqs, dtype=torch.bool)
        )

        true_times = true_time_delta_seqs[mask]
        pred_times = pred_time_delta_seqs[mask]

        return TimeValues(true_times, pred_times)


class TypeDataExtractor(TypeExtractorInterface):
    """Extracts type-related data from batch and predictions."""

    def extract_type_values(self, batch: Batch, pred: OneStepPrediction) -> TypeValues:
        """
        Extract masked type values for prediction metrics computation.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with type_predict
        """
        # Extract from Batch dataclass
        true_time_seqs = batch.time_seqs
        true_time_delta_seqs = batch.time_delta_seqs
        true_type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        pred_type_seqs = pred.type_predict

        mask = (
            batch_non_pad_mask
            if batch_non_pad_mask is not None
            else torch.ones_like(true_type_seqs, dtype=torch.bool)
        )

        true_types = true_type_seqs[mask]
        pred_types = pred_type_seqs[mask]

        return TypeValues(true_types, pred_types)


class PredictionDataExtractor(DataExtractorInterface):
    """Extracts prediction data from batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
        self.time_extractor = TimeDataExtractor()
        self.type_extractor = TypeDataExtractor()

    def extract_values(self, batch: Batch, pred: OneStepPrediction) -> MaskedValues:
        """Extract masked values for prediction metrics computation."""
        # Extract using specialized extractors
        time_values = self.time_extractor.extract_time_values(batch, pred)
        type_values = self.type_extractor.extract_type_values(batch, pred)

        return MaskedValues(
            time_values.true_times,
            type_values.true_types,
            time_values.pred_times,
            type_values.pred_types,
        )


class PredictionMetricsComputer(MetricsComputerInterface):
    """
    Computes prediction-specific metrics.

    This class focuses solely on prediction metrics computation,
    adhering to the Single Responsibility Principle. Now includes
    separate time and type metric computation methods.
    """

    def __init__(
        self,
        num_event_types: int,
        selected_metrics: Optional[List[Union[str, PredictionMetrics]]] = None,
    ):
        """
        Initialize the prediction metrics computer.

        Args:
            num_event_types: Number of event types
            data_extractor: Custom data extractor (optional, for compatibility)
            time_extractor: Custom time extractor (optional)
            type_extractor: Custom type extractor (optional)
            selected_metrics: List of metrics to compute. If None, compute all available metrics.
                             Can be strings or PredictionMetrics enum values.
        """
        self.num_event_types = num_event_types
        self._data_extractor = PredictionDataExtractor(
            num_event_types
        )
        self._time_extractor = TimeDataExtractor()
        self._type_extractor = TypeDataExtractor()

        # Process selected metrics
        if selected_metrics is None:
            # By default, compute all available metrics
            self.selected_metrics = set(self.get_available_metrics())
        else:
            # Convert to set of strings for faster lookup
            processed_metrics = []
            for metric in selected_metrics:
                if isinstance(metric, PredictionMetrics):
                    processed_metrics.append(metric.value)
                else:
                    processed_metrics.append(str(metric))

            # Validate that all selected metrics are available
            available = set(self.get_available_metrics())
            selected_set = set(processed_metrics)
            invalid_metrics = selected_set - available
            if invalid_metrics:
                logger.warning(
                    f"Invalid prediction metrics requested: {invalid_metrics}. "
                    f"Available metrics: {available}"
                )
                # Keep only valid metrics
                selected_set = selected_set & available

            self.selected_metrics = selected_set

    def compute_metrics(self, batch: Batch, pred: OneStepPrediction) -> Dict[str, Any]:
        """
        Compute selected prediction metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with dtime_predict and type_predict

        Returns:
            Dictionary of computed metrics (only selected ones)
        """
        try:
            metrics = {}
            masked = self._data_extractor.extract_values(batch, pred)

            # Compute only selected time-based metrics
            if PredictionMetrics.TIME_RMSE.value in self.selected_metrics:
                metrics["time_rmse"] = self._calculate_time_rmse(masked)
            if PredictionMetrics.TIME_MAE.value in self.selected_metrics:
                metrics["time_mae"] = self._calculate_time_mae(masked)

            # Compute selected type-based metrics only for multi-class scenarios
            if self.num_event_types > 1:
                if PredictionMetrics.TYPE_ACCURACY.value in self.selected_metrics:
                    metrics["type_accuracy"] = self._calculate_type_accuracy(masked)
                if PredictionMetrics.MACRO_F1SCORE.value in self.selected_metrics:
                    metrics["macro_f1score"] = self._calculate_f1_score(
                        masked, average="macro"
                    )
                if PredictionMetrics.RECALL.value in self.selected_metrics:
                    metrics["recall"] = self._calculate_recall(masked)
                if PredictionMetrics.PRECISION.value in self.selected_metrics:
                    metrics["precision"] = self._calculate_precision(masked)
                if PredictionMetrics.CROSS_ENTROPY.value in self.selected_metrics:
                    metrics["cross_entropy"] = self._calculate_cross_entropy(masked)
                if PredictionMetrics.CONFUSION_MATRIX.value in self.selected_metrics:
                    metrics["confusion_matrix"] = self._calculate_confusion_matrix(
                        masked
                    )

            return metrics

        except Exception as e:
            logger.error(f"Error computing prediction metrics: {e}")
            return self._get_nan_metrics()

    def compute_all_time_metrics(self, batch: Batch, pred: OneStepPrediction) -> Dict[str, Any]:
        """
        Compute all time-related metrics using the time extractor.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with dtime_predict

        Returns:
            Dictionary of computed time metrics
        """
        try:
            metrics = {}
            time_values = self._time_extractor.extract_time_values(batch, pred)

            # Compute all time metrics
            metrics["time_rmse"] = self._calculate_time_rmse_from_time_values(
                time_values
            )
            metrics["time_mae"] = self._calculate_time_mae_from_time_values(time_values)

            return metrics

        except Exception as e:
            logger.error(f"Error computing time metrics: {e}")
            return {"time_rmse": float("nan"), "time_mae": float("nan")}

    def compute_all_type_metrics(self, batch: Batch, pred: OneStepPrediction) -> Dict[str, Any]:
        """
        Compute all type-related metrics using the type extractor.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPrediction with type_predict

        Returns:
            Dictionary of computed type metrics
        """
        try:
            metrics = {}

            # Only compute type metrics for multi-class scenarios
            if self.num_event_types > 1:
                type_values = self._type_extractor.extract_type_values(batch, pred)

                # Compute all type metrics
                metrics["type_accuracy"] = (
                    self._calculate_type_accuracy_from_type_values(type_values)
                )
                metrics["macro_f1score"] = self._calculate_f1_score_from_type_values(
                    type_values, average="macro"
                )
                metrics["recall"] = self._calculate_recall_from_type_values(type_values)
                metrics["precision"] = self._calculate_precision_from_type_values(
                    type_values
                )
                metrics["cross_entropy"] = (
                    self._calculate_cross_entropy_from_type_values(type_values)
                )
                metrics["confusion_matrix"] = (
                    self._calculate_confusion_matrix_from_type_values(type_values)
                )
            else:
                # Return NaN metrics for single-class scenarios
                metrics = {
                    "type_accuracy": float("nan"),
                    "macro_f1score": float("nan"),
                    "recall": float("nan"),
                    "precision": float("nan"),
                    "cross_entropy": float("nan"),
                    "confusion_matrix": torch.full(
                        (self.num_event_types, self.num_event_types), float("nan")
                    ),
                }

            return metrics

        except Exception as e:
            logger.error(f"Error computing type metrics: {e}")
            return {
                "type_accuracy": float("nan"),
                "macro_f1score": float("nan"),
                "recall": float("nan"),
                "precision": float("nan"),
                "cross_entropy": float("nan"),
                "confusion_matrix": torch.full(
                    (self.num_event_types, self.num_event_types), float("nan")
                ),
            }

    def get_available_metrics(self) -> List[str]:
        """Get list of available prediction metrics."""
        metrics = ["time_rmse", "time_mae"]
        if self.num_event_types > 1:
            metrics.extend(
                [
                    "type_accuracy",
                    "macro_f1score",
                    "recall",
                    "precision",
                    "cross_entropy",
                    "confusion_matrix",
                ]
            )
        return metrics

    def _calculate_time_rmse(self, masked: MaskedValues) -> float:
        """Calculate time-based RMSE."""
        if masked.true_times.numel() == 0:
            return float("nan")
        return torch.sqrt(F.mse_loss(masked.pred_times, masked.true_times)).item()

    def _calculate_time_mae(self, masked: MaskedValues) -> float:
        """Calculate time-based MAE."""
        if masked.true_times.numel() == 0:
            return float("nan")
        return F.l1_loss(masked.pred_times, masked.true_times).item()

    def _calculate_time_rmse_from_time_values(self, time_values: TimeValues) -> float:
        """Calculate time-based RMSE from TimeValues."""
        if time_values.true_times.numel() == 0:
            return float("nan")
        return torch.sqrt(
            F.mse_loss(time_values.pred_times, time_values.true_times)
        ).item()

    def _calculate_time_mae_from_time_values(self, time_values: TimeValues) -> float:
        """Calculate time-based MAE from TimeValues."""
        if time_values.true_times.numel() == 0:
            return float("nan")
        return F.l1_loss(time_values.pred_times, time_values.true_times).item()

    def _calculate_type_accuracy(self, masked: MaskedValues) -> float:
        """Calculate type classification accuracy."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float("nan")
        device = masked.true_types.device
        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)
        accuracy_metric.update(masked.pred_types, masked.true_types)
        return accuracy_metric.compute().item() * 100

    def _calculate_f1_score(
        self, masked: MaskedValues, average: Literal["macro", "micro"] = "macro"
    ) -> float:
        """Calculate F1 score."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float("nan")
        device = masked.true_types.device
        f1_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_event_types, average=average
        ).to(device)
        f1_metric.update(masked.pred_types, masked.true_types)
        return f1_metric.compute().item() * 100

    def _calculate_recall(self, masked: MaskedValues) -> float:
        """Calculate recall."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float("nan")
        device = masked.true_types.device
        recall_metric = torchmetrics.Recall(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)
        recall_metric.update(masked.pred_types, masked.true_types)
        return recall_metric.compute().item() * 100

    def _calculate_precision(self, masked: MaskedValues) -> float:
        """Calculate precision."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float("nan")
        device = masked.true_types.device
        precision_metric = torchmetrics.Precision(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)
        precision_metric.update(masked.pred_types, masked.true_types)
        return precision_metric.compute().item() * 100

    def _calculate_cross_entropy(self, masked: MaskedValues) -> float:
        """Calculate cross entropy loss."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return float("nan")

        # Check format of predictions
        if masked.pred_types.dim() == 1:
            pred_logits = F.one_hot(
                masked.pred_types, num_classes=self.num_event_types
            ).float()
        else:
            pred_logits = masked.pred_types

        loss = F.cross_entropy(pred_logits, masked.true_types)
        return loss.item()

    def _calculate_confusion_matrix(self, masked: MaskedValues) -> torch.Tensor:
        """Calculate confusion matrix using torchmetrics."""
        if self.num_event_types <= 1 or masked.true_types.numel() == 0:
            return torch.full(
                (self.num_event_types, self.num_event_types), float("nan")
            )

        device = masked.true_types.device
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)
        confusion_matrix_metric.update(masked.pred_types, masked.true_types)
        return confusion_matrix_metric.compute()

    def _calculate_type_accuracy_from_type_values(
        self, type_values: TypeValues
    ) -> float:
        """Calculate type classification accuracy from TypeValues."""

        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return float("nan")
        
        device = type_values.true_types.device

        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)

        accuracy_metric.update(type_values.pred_types, type_values.true_types)

        return accuracy_metric.compute().item() * 100

    def _calculate_f1_score_from_type_values(
        self, type_values: TypeValues, average: Literal["macro", "micro"] = "macro"
    ) -> float:
        
        """Calculate F1 score from TypeValues."""
        
        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return float("nan")
        
        device = type_values.true_types.device

        f1_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_event_types, average=average
        ).to(device)

        f1_metric.update(type_values.pred_types, type_values.true_types)

        return f1_metric.compute().item() * 100

    def _calculate_recall_from_type_values(self, type_values: TypeValues) -> float:
        """Calculate recall from TypeValues."""

        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return float("nan")
        
        device = type_values.true_types.device

        recall_metric = torchmetrics.Recall(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)

        recall_metric.update(type_values.pred_types, type_values.true_types)

        return recall_metric.compute().item() * 100

    def _calculate_precision_from_type_values(self, type_values: TypeValues) -> float:
        """Calculate precision from TypeValues."""

        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return float("nan")
        
        device = type_values.true_types.device
        precision_metric = torchmetrics.Precision(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)

        precision_metric.update(type_values.pred_types, type_values.true_types)

        return precision_metric.compute().item() * 100

    def _calculate_cross_entropy_from_type_values(
        self, type_values: TypeValues
    ) -> float:
        """Calculate cross entropy loss from TypeValues."""
        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return float("nan")

        # Check format of predictions
        if type_values.pred_types.dim() == 1:
            pred_logits = F.one_hot(
                type_values.pred_types, num_classes=self.num_event_types
            ).float()
        else:
            pred_logits = type_values.pred_types

        loss = F.cross_entropy(pred_logits, type_values.true_types)
        return loss.item()

    def _calculate_confusion_matrix_from_type_values(
        self, type_values: TypeValues
    ) -> torch.Tensor:
        """Calculate confusion matrix from TypeValues using torchmetrics."""
        if self.num_event_types <= 1 or type_values.true_types.numel() == 0:
            return torch.full(
                (self.num_event_types, self.num_event_types), float("nan")
            )

        device = type_values.true_types.device
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)
        confusion_matrix_metric.update(type_values.pred_types, type_values.true_types)
        return confusion_matrix_metric.compute()

    def _get_nan_metrics(self) -> Dict[str, Any]:
        """Get a dictionary of NaN metrics for error cases."""
        metrics = {"time_rmse": float("nan"), "time_mae": float("nan")}
        if self.num_event_types > 1:
            metrics.update(
                {
                    "type_accuracy": float("nan"),
                    "macro_f1score": float("nan"),
                    "recall": float("nan"),
                    "precision": float("nan"),
                    "cross_entropy": float("nan"),
                    "confusion_matrix": torch.full(
                        (self.num_event_types, self.num_event_types), float("nan")
                    ),
                }
            )
        return metrics
