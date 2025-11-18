"""Extractors for prediction metrics.

This module contains time/type specific extractors and a combined
`PredictionDataExtractor` that provides both time and type values from a
`Batch` and a `OneStepPred`.
"""

from typing import Tuple

import torch

from new_ltpp.shared_types import Batch, OneStepPred

from .pred_types import TimeValues, TypeValues


class TimeDataExtractor:
    """Extracts time-related data from batch and predictions."""

    def extract_time_values(
        self, batch: Batch, pred_time_tensor: torch.Tensor
    ) -> TimeValues:
        """
        Extract masked time values for prediction metrics computation.

        Args:
            batch: Batch object containing ground truth sequences
            pred_time_tensor: Tensor of predicted time deltas
        """
        true_time_delta_seqs = batch.time_delta_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        true_times = true_time_delta_seqs[batch_non_pad_mask]
        pred_times = pred_time_tensor[batch_non_pad_mask]

        return TimeValues(true_times=true_times, pred_times=pred_times)


class TypeDataExtractor:
    """Extracts type-related data from batch and predictions."""

    def extract_type_values(
        self, batch: Batch, pred_type_tensor: torch.Tensor
    ) -> TypeValues:
        """
        Extract masked type values for prediction metrics computation.

        Args:
            batch: Batch object containing ground truth sequences
            pred_type_tensor: Tensor of predicted event types
        """
        true_type_seqs = batch.type_seqs
        batch_non_pad_mask = batch.seq_non_pad_mask

        true_types = true_type_seqs[batch_non_pad_mask]
        pred_types = pred_type_tensor[batch_non_pad_mask]

        return TypeValues(true_types=true_types, pred_types=pred_types)


class PredictionDataExtractor:
    """Extracts prediction data from batch and predictions."""

    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
        self.time_extractor = TimeDataExtractor()
        self.type_extractor = TypeDataExtractor()

    def extract_values(
        self, batch: Batch, pred: OneStepPred
    ) -> tuple[TimeValues, TypeValues]:
        """Extract time and type values as a tuple.

        Args:
            batch: Batch object containing ground truth sequences
            pred: OneStepPred with dtime_predict and type_predict

        Returns:
            Tuple of (TimeValues, TypeValues)
        """

        pred_time_tensor = pred["dtime_predict"]
        pred_type_tensor = pred["type_predict"]

        time_values = self.time_extractor.extract_time_values(batch, pred_time_tensor)
        type_values = self.type_extractor.extract_type_values(batch, pred_type_tensor)

        return (time_values, type_values)
