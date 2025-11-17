from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TypedDict, Tuple, Any

import torch



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


@dataclass
class Batch:
	"""Container for a minibatch used across the project.
	Args:
		time_seqs: Tensor of shape (batch_size, seq_len)
		time_delta_seqs: Tensor of shape (batch_size, seq_len)
		type_seqs: Tensor of shape (batch_size, seq_len)
		seq_non_pad_mask: Boolean tensor of shape (batch_size, seq_len)
	"""

	time_seqs: torch.Tensor
	time_delta_seqs: torch.Tensor
	type_seqs: torch.Tensor
	seq_non_pad_mask: torch.Tensor

	def __post_init__(self):
		"""Validate and normalize batch after initialization."""
		# Ensure all required tensors have the same batch size
		batch_size = self.time_seqs.shape[0]
		assert self.time_delta_seqs.shape[0] == batch_size, "time_delta_seqs batch size mismatch"
		assert self.type_seqs.shape[0] == batch_size, "type_seqs batch size mismatch"
		assert self.seq_non_pad_mask.shape[0] == batch_size, "seq_non_pad_mask batch size mismatch"

	@classmethod
	def from_mapping(cls, mapping: Dict[str, torch.Tensor]) -> "Batch":
		"""Create a `Batch` from a mapping like the tokenizer's output.

        The mapping is expected to contain the keys used by
        `EventTokenizer.model_input_names`:
        `time_seqs`, `time_delta_seqs`, `type_seqs`, `seq_non_pad_mask`.
		"""
		return cls(
			time_seqs=mapping["time_seqs"],
			time_delta_seqs=mapping["time_delta_seqs"],
			type_seqs=mapping["type_seqs"],
			seq_non_pad_mask=mapping["seq_non_pad_mask"],
		)

	def to_mapping(self) -> Dict[str, Any]:
		"""Return a plain mapping usable by other components or for serialization."""
		return {
			"time_seqs": self.time_seqs,
			"time_delta_seqs": self.time_delta_seqs,
			"type_seqs": self.type_seqs,
			"seq_non_pad_mask": self.seq_non_pad_mask,
        }


	def to_device(self, device: torch.device) -> "Batch":
		"""Move tensor fields to `device` in-place and return self.

		Applies `.to(device)` to each tensor field.
		"""
		self.time_seqs = self.time_seqs.to(device)
		self.time_delta_seqs = self.time_delta_seqs.to(device)
		self.type_seqs = self.type_seqs.to(device)
		self.seq_non_pad_mask = self.seq_non_pad_mask.to(device)
		return self
      
class DataStats(TypedDict):
    """Container for dataset statistics used in model configuration.

    Args:
        num_event_types: Number of unique event types in the dataset
        end_time_max: Maximum end time value in the dataset
        dtime_max: Maximum time delta value in the dataset
    """

    num_event_types: int
    end_time_max: float
    dtime_max: float

class OneStepPred(TypedDict):
    """Container for one-step-ahead predictions.

    Args:
        dtime_predict: Predicted time deltas for the next event
        type_predict: Predicted event types for the next event
    """
    dtime_predict: torch.Tensor
    type_predict: torch.Tensor

@dataclass
class SimulationResult:
    """Container for simulated sequences with time deltas.

    Args:
        time_seqs: Simulated event times
		dtime_seqs: Simulated time deltas
        type_seqs: Simulated event types
    """
    time_seqs: torch.Tensor
    dtime_seqs: torch.Tensor
    type_seqs: torch.Tensor
    mask: torch.Tensor
    
    def __post_init__(self):
        """Validate simulation result after initialization."""
        # Ensure all tensors have the same shape
        assert self.time_seqs.shape == self.type_seqs.shape, "time_seqs and type_seqs must have the same shape"
        assert self.time_seqs.shape == self.dtime_seqs.shape, "dtime_seqs and time_seqs must have the same shape"
        assert self.time_seqs.shape == self.mask.shape, "mask and time_seqs must have the same shape"

    def mask_values(self, mask_id = -1) -> None:
        
        """Get the valid (non-masked) time values, flattened.
        
        Returns:
            1D tensor of valid time values
        """

        self.time_seqs[self.mask] = mask_id
        self.dtime_seqs[self.mask] = mask_id
        self.type_seqs[self.mask] = mask_id

        return 
