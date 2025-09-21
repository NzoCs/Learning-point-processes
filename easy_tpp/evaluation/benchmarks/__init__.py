"""
Benchmarks module for evaluating TPP models.

This module contains simple benchmark implementations that provide baseline
performance comparisons for temporal point process models.
"""

from .base_bench import BaseBenchmark, BenchmarkMode
from .benchmark_manager import BenchmarkManager, Benchmarks
from .last_mark_bench import LastMarkBenchmark
from .mean_bench import MeanInterTimeBenchmark
from .sample_distrib_intertime_bench import (
    InterTimeDistributionBenchmark,
)
from .sample_distrib_mark_bench import (
    MarkDistributionBenchmark,
)

__all__ = [
    "BaseBenchmark",
    "BenchmarkMode",
    "run_benchmark",
    "MeanInterTimeBenchmark",
    "InterTimeDistributionBenchmark",
    "MarkDistributionBenchmark",
    "LastMarkBenchmark",
    "run_mean_benchmark",
    "run_intertime_distribution_benchmark",
    "run_mark_distribution_benchmark",
    "run_last_mark_benchmark",
]
