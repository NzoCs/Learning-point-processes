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
	attention_mask: torch.Tensor  # Can be an empty list

	# Optional additional masks/features
	type_mask: Optional[torch.Tensor] = None


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
    """Container for simulated sequences.

    Args:
        time_seqs: Simulated event times
        type_seqs: Simulated event types
    """
    time_seqs: torch.Tensor
    type_seqs: torch.Tensor