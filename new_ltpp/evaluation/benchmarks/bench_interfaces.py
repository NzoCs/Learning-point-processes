from __future__ import annotations

from typing import Any, Dict, Iterable, Protocol, Tuple

import torch


class BenchmarkInterface(Protocol):
    """Typing protocol describing the minimal benchmark contract.

    This mirrors the public surface used by other modules: a `benchmark_name`
    property, a `_prepare_benchmark` method, prediction creation methods and
    an `evaluate` method that returns a dictionary of results.
    """

    @property
    def benchmark_name(self) -> str:  # pragma: no cover - typing only
        ...

    def _prepare_benchmark(self) -> None:  # pragma: no cover - typing only
        ...

    def _create_predictions(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - typing only
        ...

    def _create_time_predictions(self, batch: Tuple) -> torch.Tensor:  # pragma: no cover - typing only
        ...

    def _create_type_predictions(self, batch: Tuple) -> torch.Tensor:  # pragma: no cover - typing only
        ...

    def evaluate(self) -> Dict[str, Any]:  # pragma: no cover - typing only
        ...
