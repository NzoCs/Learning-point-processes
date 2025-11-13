from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any

import torch



@dataclass
class TPPSequence:
    """A single temporal point process sequence.
    
    Args:
        time_seqs: Sequence of event times
        time_delta_seqs: Sequence of time deltas between events
        type_seqs: Sequence of event types
    """
    time_seqs: List[float]
    time_delta_seqs: List[float]
    type_seqs: List[int]

    def to_dict(self) -> Dict[str, Union[List[float], List[int]]]:
        """Convert to dictionary format for compatibility with existing code."""
        return {
            "time_seqs": self.time_seqs,
            "time_delta_seqs": self.time_delta_seqs,
            "type_seqs": self.type_seqs,
        }


# ============================================================================
# BATCH DATA STRUCTURE
# ============================================================================

@dataclass
class Batch:
	"""Container for a minibatch used across the project.

	Fields correspond to keys produced by `EventTokenizer.pad` / `_pad`.

	Notes:
	- Types are `Any` to avoid hard dependency on a particular tensor
	  library; call sites can convert to torch/np as needed.
	- `attention_mask` may be an empty list when the tokenizer did not
	  produce it.
	"""

	time_seqs: torch.Tensor
	time_delta_seqs: torch.Tensor
	type_seqs: torch.Tensor
	seq_non_pad_mask: torch.Tensor
	attention_mask: Optional[Union[torch.Tensor, list]] = None  # Can be an empty list or None

	# Optional additional masks/features
	type_mask: Optional[torch.Tensor] = None

	def __post_init__(self):
		"""Validate and normalize batch after initialization."""
		# Handle empty or None attention_mask
		if self.attention_mask is None or (isinstance(self.attention_mask, (list, tuple)) and len(self.attention_mask) == 0):
			self.attention_mask = torch.empty(0, device=self.time_seqs.device)
		
		# Ensure all required tensors have the same batch size
		batch_size = self.time_seqs.shape[0]
		assert self.time_delta_seqs.shape[0] == batch_size, "time_delta_seqs batch size mismatch"
		assert self.type_seqs.shape[0] == batch_size, "type_seqs batch size mismatch"
		assert self.seq_non_pad_mask.shape[0] == batch_size, "seq_non_pad_mask batch size mismatch"
		
		if isinstance(self.attention_mask, torch.Tensor) and self.attention_mask.numel() > 0:
			assert self.attention_mask.shape[0] == batch_size, "attention_mask batch size mismatch"

	@classmethod
	def from_mapping(cls, mapping: Dict[str, torch.Tensor]) -> "Batch":
		"""Create a `Batch` from a mapping like the tokenizer's output.

		The mapping is expected to contain the keys used by
		`EventTokenizer.model_input_names`:
		`time_seqs`, `time_delta_seqs`, `type_seqs`, `seq_non_pad_mask`,
		`attention_mask`.
		"""
		return cls(
			time_seqs=mapping["time_seqs"],
			time_delta_seqs=mapping["time_delta_seqs"],
			type_seqs=mapping["type_seqs"],
			seq_non_pad_mask=mapping["seq_non_pad_mask"],
			attention_mask=mapping["attention_mask"],
			type_mask=mapping.get("type_mask"),
		)

	def to_mapping(self) -> Dict[str, Any]:
		"""Return a plain mapping usable by other components or for serialization."""
		return {
			"time_seqs": self.time_seqs,
			"time_delta_seqs": self.time_delta_seqs,
			"type_seqs": self.type_seqs,
			"seq_non_pad_mask": self.seq_non_pad_mask,
			"attention_mask": self.attention_mask,
			"type_mask": self.type_mask,
        }


	def to_device(self, device: torch.device) -> "Batch":
		"""Move tensor fields to `device` in-place and return self.

		Applies `.to(device)` to each tensor field.
		"""
		self.time_seqs = self.time_seqs.to(device)
		self.time_delta_seqs = self.time_delta_seqs.to(device)
		self.type_seqs = self.type_seqs.to(device)
		self.seq_non_pad_mask = self.seq_non_pad_mask.to(device)
		if self.attention_mask not in ([], None):
			self.attention_mask = self.attention_mask.to(device)
		if self.type_mask is not None:
			self.type_mask = self.type_mask.to(device)

		return self

@dataclass
class OneStepPrediction:
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
        type_seqs: Simulated event types
        mask: Mask indicating valid events in the sequences
    """
    time_seqs: torch.Tensor
    dtime_seqs: torch.Tensor
    type_seqs: torch.Tensor
    
    def __post_init__(self):
        """Validate simulation result after initialization."""
        # Ensure all tensors have the same shape
        assert self.time_seqs.shape == self.type_seqs.shape, "time_seqs and type_seqs must have the same shape"
        assert self.time_seqs.shape == self.dtime_seqs.shape, "dtime_seqs and time_seqs must have the same shape"
    
    @classmethod
    def from_simulation_tensors(
        cls,
        time_seqs: torch.Tensor,
        dtime_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        mask: torch.Tensor
    ) -> SimulationResult:
        """Convert batched simulation tensors to a SimulationResult with masked tensors.
        
        Args:
            time_seqs: Tensor of shape (batch_size, seq_len) with event times
            dtime_seqs: Tensor of shape (batch_size, seq_len) with time deltas
            type_seqs: Tensor of shape (batch_size, seq_len) with event types
            mask: Tensor of shape (batch_size, seq_len) indicating valid events
            
        Returns:
            SimulationResult with masked tensors moved to CPU
        """
        # Apply mask directly and move to CPU
        return cls(
            time_seqs=(time_seqs * mask).detach().cpu(),
            dtime_seqs=(dtime_seqs * mask).detach().cpu(),
            type_seqs=(type_seqs * mask).detach().cpu()
        )
