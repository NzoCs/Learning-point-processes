"""
Benchmarks module for evaluating TPP models.

This module contains simple benchmark implementations that provide baseline
performance comparisons for temporal point process models.
"""

from .base_bench import Benchmark
from .benchmark_manager import BenchmarkManager, BenchmarksEnum
from .last_mark_bench import LastMarkBenchmark
from .mean_bench import MeanInterTimeBenchmark
from .sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from .sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)
from .time_bench import TimeBenchmark
from .type_bench import TypeBenchmark

__all__ = [
    "Benchmark",
    "TimeBenchmark",
    "TypeBenchmark",
    "BenchmarkManager",
    "BenchmarksEnum",
    "MeanInterTimeBenchmark",
    "InterTimeDistributionBenchmark",
    "MarkDistributionBenchmark",
    "LastMarkBenchmark",
]
