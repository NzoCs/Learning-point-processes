"""Protocol for benchmarks.

Provides IBenchmark and related Protocols for IDE type checking + isinstance() support.
Complements existing BaseBenchmark ABC.
"""

from typing import Any, Dict, Protocol, runtime_checkable

import torch

from new_ltpp.shared_types import Batch


@runtime_checkable
class IBenchmark(Protocol):
    """Protocol for IDE type checking + isinstance() support."""

    @property
    def benchmark_name(self) -> str:
        """Return the name of this benchmark."""
        ...

    def evaluate(self) -> Dict[str, Any]:
        """Run the benchmark evaluation."""
        ...


@runtime_checkable
class ITimeBenchmark(Protocol):
    """Protocol for time prediction benchmarks."""

    def _create_dtime_predictions(self, batch: Batch) -> torch.Tensor:
        """Create time predictions for a given batch."""
        ...


@runtime_checkable
class ITypeBenchmark(Protocol):
    """Protocol for type/mark prediction benchmarks."""

    def _create_type_predictions(self, batch: Batch) -> torch.Tensor:
        """Create type predictions for a given batch."""
        ...
