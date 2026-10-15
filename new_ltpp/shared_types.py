from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NotRequired, TypedDict

import torch

STOPPED_OBSERVATION_PROTOCOL = "complete_stopped_event_budget_v1"


def uses_stopped_event_budget(model_name: str) -> bool:
    return model_name in ("FullyNN", "RMTPP")


class TPPSequence(TypedDict):
    """A single temporal point process sequence.

    Args:
        time_seqs: Sequence of event times
        time_delta_seqs: Sequence of time deltas between events
        type_seqs: Sequence of event types
    """

    time_seqs: List[float]
    time_delta_seqs: List[float]
    type_seqs: List[int]
    sequence_id: NotRequired[str]
    no_next_event: NotRequired[bool]


@dataclass
class Batch:
    """Container for a minibatch used across the project.

    Also used as SimulationResult (SimulationResult = Batch).

    Args:
            time_seqs: Tensor of shape (batch_size, seq_len)
            time_delta_seqs: Tensor of shape (batch_size, seq_len)
            type_seqs: Tensor of shape (batch_size, seq_len)
            valid_event_mask: Boolean tensor of shape (batch_size, seq_len).
                              True indicates valid events (non-padded or valid simulation).
    """

    time_seqs: torch.Tensor
    time_delta_seqs: torch.Tensor
    type_seqs: torch.Tensor
    valid_event_mask: torch.Tensor
    sequence_ids: List[str] | None = None
    no_next_event_mask: torch.Tensor | None = None

    def __post_init__(self):
        """Validate and normalize batch after initialization."""
        # Ensure all required tensors have the same batch size
        batch_size = self.time_seqs.shape[0]
        if self.sequence_ids is not None and len(self.sequence_ids) != batch_size:
            raise ValueError("sequence_ids must match the batch rows")
        assert (
            self.time_delta_seqs.shape[0] == batch_size
        ), "time_delta_seqs batch size mismatch"
        assert self.type_seqs.shape[0] == batch_size, "type_seqs batch size mismatch"
        assert (
            self.valid_event_mask.shape[0] == batch_size
        ), "valid_event_mask batch size mismatch"

    @classmethod
    def from_mapping(cls, mapping: Dict[str, torch.Tensor]) -> "Batch":
        """Create a `Batch` from a mapping like the tokenizer's output.

        The mapping is expected to contain the keys used by
        `EventTokenizer.model_input_names`:
        `time_seqs`, `time_delta_seqs`, `type_seqs`, `valid_event_mask`.
        """
        return cls(
            time_seqs=mapping["time_seqs"],
            time_delta_seqs=mapping["time_delta_seqs"],
            type_seqs=mapping["type_seqs"],
            valid_event_mask=mapping["valid_event_mask"],
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Return a plain mapping usable by other components or for serialization."""
        return {
            "time_seqs": self.time_seqs,
            "time_delta_seqs": self.time_delta_seqs,
            "type_seqs": self.type_seqs,
            "valid_event_mask": self.valid_event_mask,
        }

    def to_device(self, device: torch.device) -> "Batch":
        """Move tensor fields to `device` in-place and return self.

        Applies `.to(device)` to each tensor field.
        """
        self.time_seqs = self.time_seqs.to(device)
        self.time_delta_seqs = self.time_delta_seqs.to(device)
        self.type_seqs = self.type_seqs.to(device)
        self.valid_event_mask = self.valid_event_mask.to(device)
        if self.no_next_event_mask is not None:
            self.no_next_event_mask = self.no_next_event_mask.to(device)
        return self


class DataInfo(TypedDict):
    """Container for dataset statistics used in model configuration.

    Args:
        num_event_types: Number of unique event types in the dataset
        end_time_max: Maximum end time value in the dataset
        dtime_max: Maximum time delta value in the dataset
        pad_token_id: Padding token ID used in the dataset
    """

    num_event_types: int
    end_time_max: float
    dtime_max: float
    pad_token_id: int


class OneStepPred(TypedDict):
    """Container for one-step-ahead predictions.

    Args:
        dtime_predict: Predicted time deltas for the next event
        type_predict: Predicted event types for the next event
    """

    dtime_predict: torch.Tensor
    type_predict: torch.Tensor
    no_event_probability: NotRequired[torch.Tensor]


# Backward compatibility alias
SimulationResult = Batch
