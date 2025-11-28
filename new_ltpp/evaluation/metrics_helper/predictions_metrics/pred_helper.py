from collections.abc import Callable
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
import torchmetrics

from new_ltpp.shared_types import Batch, OneStepPred
from new_ltpp.utils import logger

from .pred_extractor import (
    PredictionDataExtractor,
    TimeDataExtractor,
    TypeDataExtractor,
)
from .pred_types import PredMetrics, TimeValues, TypeValues


class PredMetricsHelper:
    """
    Computes prediction-specific metrics.

    This class focuses solely on prediction metrics computation,
    adhering to the Single Responsibility Principle. Now includes
    separate time and type metric computation methods.
    """

    def __init__(
        self,
        num_event_types: int,
        selected_metrics: Optional[List[Union[str, PredMetrics]]] = None,
    ):
        """
        Initialize the prediction metrics computer.

        Args:
            num_event_types: Number of event types
            data_extractor: Custom data extractor (optional, for compatibility)
            time_extractor: Custom time extractor (optional)
            type_extractor: Custom type extractor (optional)
            selected_metrics: List of metrics to compute. If None, compute all available metrics.
                             Can be strings or PredMetrics enum values.
        """
        self.num_event_types = num_event_types
        self._data_extractor = PredictionDataExtractor(num_event_types)
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
                if isinstance(metric, PredMetrics):
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

    def compute_metrics(self, batch: Batch, pred: OneStepPred) -> Dict[str, Any]:
        """
        Compute selected prediction metrics.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPred with dtime_predict and type_predict

        Returns:
            Dictionary of computed metrics (only selected ones)
        """

        metrics = {}
        time_values, type_values = self._data_extractor.extract_values(batch, pred)

        time_metric_mapping = self._build_time_metric_mapping(time_values)
        for metric_name, entry in time_metric_mapping.items():
            func, args = entry
            if metric_name in self.selected_metrics:
                metrics[metric_name] = func(*args)

        if self.num_event_types > 1:
            type_metric_mapping = self._build_type_metric_mapping(type_values)
            for metric_name, entry in type_metric_mapping.items():
                func, args = entry
                if metric_name in self.selected_metrics:
                    metrics[metric_name] = func(*args)

        return metrics

    def compute_all_time_metrics(
        self, batch: Batch, pred_time_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        
        """
        Compute all time-related metrics using the time extractor.

        Args:
            batch: Batch object containing ground truth sequences
            pred_time_tensor: Tensor of predicted times

        Returns:
            Dictionary of computed time metrics
        """

        metrics = {}
        time_values = self._time_extractor.extract_time_values(batch, pred_time_tensor)
        metric_mapping = self._build_time_metric_mapping(time_values)
        for metric_name, entry in metric_mapping.items():
            func, args = entry
            metrics[metric_name] = func(*args)
        return metrics

    def compute_all_type_metrics(
        self, batch: Batch, pred_type_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute all type-related metrics using the type extractor.

        Args:
            batch: Batch object containing ground truth sequences
            pred_type_tensor: Tensor of predicted types

        Returns:
            Dictionary of computed type metrics
        """
        metrics = {}

        if self.num_event_types > 1:
            type_values = self._type_extractor.extract_type_values(
                batch, pred_type_tensor
            )
            metric_mapping = self._build_type_metric_mapping(type_values)
            for metric_name, entry in metric_mapping.items():
                func, args = entry
                metrics[metric_name] = func(*args)
        else:
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

    def _calculate_time_rmse(self, time_values: TimeValues) -> float:
        """Calculate time-based RMSE."""
        if time_values["true_times"].numel() == 0:
            return float("nan")
        return torch.sqrt(
            F.mse_loss(time_values["pred_times"], time_values["true_times"])
        ).item()

    def _calculate_time_mae(self, time_values: TimeValues) -> float:
        """Calculate time-based MAE."""
        if time_values["true_times"].numel() == 0:
            return float("nan")
        return F.l1_loss(time_values["pred_times"], time_values["true_times"]).item()

    def _calculate_type_accuracy(self, type_values: TypeValues) -> float:
        """Calculate type classification accuracy."""

        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return float("nan")

        device = type_values["true_types"].device

        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)

        accuracy_metric.update(type_values["pred_types"], type_values["true_types"])

        return accuracy_metric.compute().item() * 100

    def _calculate_f1_score(
        self, type_values: TypeValues, average: Literal["macro", "micro"] = "macro"
    ) -> float:
        """Calculate F1 score."""

        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return float("nan")

        device = type_values["true_types"].device

        f1_metric = torchmetrics.F1Score(
            task="multiclass", num_classes=self.num_event_types, average=average
        ).to(device)

        f1_metric.update(type_values["pred_types"], type_values["true_types"])

        return f1_metric.compute().item() * 100

    def _calculate_recall(self, type_values: TypeValues) -> float:
        """Calculate recall."""

        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return float("nan")

        device = type_values["true_types"].device

        recall_metric = torchmetrics.Recall(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)

        recall_metric.update(type_values["pred_types"], type_values["true_types"])  # type: ignore

        return recall_metric.compute().item() * 100

    def _calculate_precision(self, type_values: TypeValues) -> float:
        """Calculate precision."""

        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return float("nan")

        device = type_values["true_types"].device
        precision_metric = torchmetrics.Precision(
            task="multiclass", num_classes=self.num_event_types, average="macro"
        ).to(device)

        precision_metric.update(type_values["pred_types"], type_values["true_types"])

        return precision_metric.compute().item() * 100

    def _calculate_cross_entropy(self, type_values: TypeValues) -> float:
        """Calculate cross entropy loss."""
        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return float("nan")

        # Check format of predictions
        if type_values["pred_types"].dim() == 1:
            pred_logits = F.one_hot(
                type_values["pred_types"], num_classes=self.num_event_types
            ).float()
        else:
            pred_logits = type_values["pred_types"]

        loss = F.cross_entropy(pred_logits, type_values["true_types"])
        return loss.item()

    def _calculate_confusion_matrix(self, type_values: TypeValues) -> torch.Tensor:
        """Calculate confusion matrix using torchmetrics."""
        if self.num_event_types <= 1 or type_values["true_types"].numel() == 0:
            return torch.full(
                (self.num_event_types, self.num_event_types), float("nan")
            )

        device = type_values["true_types"].device
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_event_types
        ).to(device)
        confusion_matrix_metric.update(
            type_values["pred_types"], type_values["true_types"]
        )
        return confusion_matrix_metric.compute()

    def _get_nan_metrics(self) -> Dict[str, Any]:
        """Get a dictionary of NaN metrics for error cases."""
        metrics: Dict[str, Any] = {"time_rmse": float("nan"), "time_mae": float("nan")}
        if self.num_event_types <= 1:
            return metrics

        type_nan_metrics: Dict[str, Any] = {
            "type_accuracy": float("nan"),
            "macro_f1score": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "cross_entropy": float("nan"),
            "confusion_matrix": torch.full(
                (self.num_event_types, self.num_event_types), float("nan")
            ),
        }

        metrics = {**metrics, **type_nan_metrics}
        return metrics

    def _build_time_metric_mapping(
        self, time_values: TimeValues
    ) -> Dict[str, tuple[Callable[..., Any], tuple[Any, ...]]]:
        return {
            PredMetrics.TIME_RMSE.value: (self._calculate_time_rmse, (time_values,)),
            PredMetrics.TIME_MAE.value: (self._calculate_time_mae, (time_values,)),
        }

    def _build_type_metric_mapping(
        self, type_values: TypeValues
    ) -> Dict[str, tuple[Callable[..., Any], tuple[Any, ...]]]:
        return {
            PredMetrics.TYPE_ACCURACY.value: (
                self._calculate_type_accuracy,
                (type_values,),
            ),
            PredMetrics.MACRO_F1SCORE.value: (
                self._calculate_f1_score,
                (type_values, "macro"),
            ),
            PredMetrics.RECALL.value: (
                self._calculate_recall,
                (type_values,),
            ),
            PredMetrics.PRECISION.value: (
                self._calculate_precision,
                (type_values,),
            ),
            PredMetrics.CROSS_ENTROPY.value: (
                self._calculate_cross_entropy,
                (type_values,),
            ),
            PredMetrics.CONFUSION_MATRIX.value: (
                self._calculate_confusion_matrix,
                (type_values,),
            ),
        }
