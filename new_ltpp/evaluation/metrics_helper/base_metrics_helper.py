from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from new_ltpp.utils import logger


class MetricsHelper(ABC):
    """Base class for metric computation helpers.

    Responsibilities:
    - store `num_event_types`
    - normalise and validate `selected_metrics`
    - expose an abstract `compute_metrics` API subclasses must implement
    """

    def __init__(
        self,
        num_event_types: int,
        selected_metrics: Optional[List[Union[str, Any]]] = None,
    ) -> None:
        self.num_event_types = num_event_types

        if selected_metrics is None:
            self.selected_metrics: Set[str] = set(self.get_available_metrics())
        else:
            processed: List[str] = []
            for m in selected_metrics:
                # support enums or objects with `value`, otherwise str()
                try:
                    processed.append(m.value)  # type: ignore[attr-defined]
                except Exception:
                    processed.append(str(m))

            available = set(self.get_available_metrics())
            invalid = set(processed) - available
            if invalid:
                logger.warning(
                    f"Requested metrics not available: {sorted(list(invalid))}. "
                    f"Available: {sorted(list(available))}"
                )
            # keep only valid metrics
            self.selected_metrics = set(processed) & available

    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Return list of metric names available from this helper."""

    @abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Compute and return a mapping metric_name -> value.

        Subclasses should compute all metrics they support (or only the
        requested subset) and return a dict. The `selected_metrics` set can
        be used by subclasses to limit computation.
        """

    def _filter_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Return a filtered mapping containing only `selected_metrics`.

        Useful for subclasses which build a full mapping of metric_name ->
        (callable, args) and want to return only the selected entries.
        """
        if not self.selected_metrics:
            return mapping
        return {k: v for k, v in mapping.items() if k in self.selected_metrics}
