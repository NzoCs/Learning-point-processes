"""
Prediction metrics computation class.
"""

from typing import Any, Dict, List, Optional, Union, Literal

import torch
import torch.nn.functional as F
import torchmetrics

from new_ltpp.utils import logger

from new_ltpp.evaluation.metrics_helper.metrics_interfaces import (
    DataExtractorInterface,
    MetricsComputerInterface,
    TimeExtractorInterface,
    TypeExtractorInterface,
)
from new_ltpp.evaluation.metrics_helper.shared_types import (
    MaskedValues, PredictionMetrics, TimeValues, TypeValues
)


class TimeDataExtractor(TimeExtractorInterface):
    """Extracts time-related data from batch and predictions."""

    def extract_time_values(self, batch: Any, pred: Any) -> TimeValues:
        """
        Extract masked time values for prediction metrics computation.

        Args:
            batch: Input batch data
            pred: Either a tuple/list of predictions (pred[0] = time predictions)
                  or just the time predictions tensor directly
        """
        # Extraction according to new_ltpp format
        if len(batch) >= 6:
            (
                true_time_seqs,
                true_time_delta_seqs,
                true_type_seqs,
                batch_non_pad_mask,
                attention_mask,
                _,
            ) = batch
        elif len(batch) >= 5:
            (
                true_time_seqs,
                true_time_delta_seqs,
                true_type_seqs,
                batch_non_pad_mask,
                attention_mask,
            ) = batch
        else:
            raise ValueError(
                "Batch values must contain at least 5 elements for prediction mode."
            )

        # Handle both cases: pred as tuple/list or pred as direct time tensor
        if isinstance(pred, (tuple, list)) and len(pred) > 0:
            # pred is a tuple/list, extract time predictions from index 0
            pred_time_delta_seqs = pred[0]
        else:
            # pred is directly the time predictions tensor
            pred_time_delta_seqs = pred

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

    def extract_type_values(self, batch: Any, pred: Any) -> TypeValues:
        """
        Extract masked type values for prediction metrics computation.

        Args:
            batch: Input batch data
            pred: Either a tuple/list of predictions (pred[1] = type predictions)
                  or just the type predictions tensor directly
        """
        # Extraction according to new_ltpp format
        if len(batch) >= 6:
            (
                true_time_seqs,
                true_time_delta_seqs,
                true_type_seqs,
                batch_non_pad_mask,
                attention_mask,
                _,
            ) = batch
        elif len(batch) >= 5:
            (
                true_time_seqs,
                true_time_delta_seqs,
                true_type_seqs,
                batch_non_pad_mask,
                attention_mask,
            ) = batch
        else:
            raise ValueError(
                "Batch values must contain at least 5 elements for prediction mode."
            )

        # Handle both cases: pred as tuple/list or pred as direct type tensor
        if isinstance(pred, (tuple, list)) and len(pred) > 1:
            # pred is a tuple/list, extract type predictions from index 1
            pred_type_seqs = pred[1]
        else:
            # pred is directly the type predictions tensor
            pred_type_seqs = pred

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

    def extract_values(self, batch: Any, pred: Any) -> MaskedValues:
        """Extract masked values for prediction metrics computation."""
        # Add debug logging to understand the data structure
        logger.debug(
            f"DEBUG: Batch type: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
        )
        logger.debug(
            f"DEBUG: Pred type: {type(pred)}, length: {len(pred) if hasattr(pred, '__len__') else 'N/A'}"
        )

        # Check if batch elements are strings (which would cause the error)
        for i, item in enumerate(batch):
            logger.debug(f"DEBUG: Batch[{i}] type: {type(item)}")
            if isinstance(item, str):
                logger.error(f"DEBUG: Found string in batch at index {i}: {item}")

        # Check pred elements too
        for i, item in enumerate(pred):
            logger.debug(f"DEBUG: Pred[{i}] type: {type(item)}")
            if isinstance(item, str):
                logger.error(f"DEBUG: Found string in pred at index {i}: {item}")

        # Extract using specialized extractors
        time_values = self.time_extractor.extract_time_values(batch, pred)
        type_values = self.type_extractor.extract_type_values(batch, pred)

        return MaskedValues(
            time_values.true_times,
            type_values.true_types,
            time_values.pred_times,
            type_values.pred_types,
        )
