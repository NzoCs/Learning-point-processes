"""Types shared across the project.

This module defines data structures used throughout the project:
- `Batch`: Container for batched event sequences
- `PaddingStrategy`: Enum for padding strategies
- `TruncationStrategy`: Enum for truncation strategies

Batch Format:
-------------
Typical shapes produced by `EventTokenizer` (numpy arrays):
- `time_seqs`: float32, shape (batch_size, seq_len)
- `time_delta_seqs`: float32, shape (batch_size, seq_len)
- `type_seqs`: int64, shape (batch_size, seq_len)
- `seq_non_pad_mask`: bool, shape (batch_size, seq_len)
- `attention_mask`: bool, shape (batch_size, seq_len, seq_len) or an empty list when unused
- `type_mask` (optional): int32, shape (batch_size, seq_len, num_event_types)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union, List

import torch


# ============================================================================
# STRATEGY ENUMS
# ============================================================================

class ExplicitEnum(str, Enum):
    """Enum with more explicit error message for missing values."""

    def __str__(self):
        return str(self.value)

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """Padding strategies for event sequences.
    
    Values:
        LONGEST: Pad to the longest sequence in the batch
        MAX_LENGTH: Pad to a specified maximum length
        DO_NOT_PAD: Do not apply padding
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(ExplicitEnum):
    """Truncation strategies for event sequences.
    
    Values:
        LONGEST_FIRST: Truncate the longest sequences first
        DO_NOT_TRUNCATE: Do not apply truncation
    """

    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


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

    def to_dict(self) -> Dict[str, List]:
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

__all__ = ["Batch", "PaddingStrategy", "TruncationStrategy"]

