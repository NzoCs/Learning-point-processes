from typing import TYPE_CHECKING

from ..point_process_kernels.factory import create_point_process_kernel
from .base_test import ITest
from .mmd_test import MMDTwoSampleTest

if TYPE_CHECKING:
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig
    from new_ltpp.configs.window_test_config import WindowTestConfig

    from .stein_test import SteinTest
    from .window_tests import FixedWindowMMDTest, PredictiveLossTest


def create_statistical_test(
    config: "StatisticalTestConfig | WindowTestConfig", *, reference=None
) -> "ITest | FixedWindowMMDTest | PredictiveLossTest | SteinTest":
    """Factory function to create a statistical test based on the config.

    Args:
        config: The statistical test configuration.

    Fixed-window tests expose compute_statistics on explicit windows. Stein is
    one-sample and deliberately does not implement the two-sample ITest protocol.
    """
    from .window_tests import (
        CountRBFKernel,
        FixedWindowMMDTest,
        FixedWindowSignatureKernel,
        PredictiveLossTest,
    )

    if config.test_type == "window_mmd":
        if config.point_process_kernel == "count_rbf":
            return FixedWindowMMDTest(
                CountRBFKernel(
                    horizon=config.horizon,
                    num_event_types=config.num_event_types,
                    count_scale=config.count_scale,
                    sigma=config.sigma,
                ),
                n_samples=config.n_samples,
                seed=config.seed,
            )
        kernel = FixedWindowSignatureKernel(
            horizon=config.horizon,
            num_event_types=config.num_event_types,
            representation=config.representation,
            grid=config.grid,
            count_scale=config.count_scale,
            gap_scale=config.gap_scale,
            rank_scale=config.rank_scale,
            sigma=config.sigma,
            dyadic_order=config.dyadic_order,
            max_batch=config.max_batch,
            space_kernel_type=config.space_kernel_type,
        )
        return FixedWindowMMDTest(kernel, n_samples=config.n_samples, seed=config.seed)
    if config.test_type in {"relative_loss", "ksd"}:
        if reference is None or reference.num_event_types != config.num_event_types:
            raise ValueError("A matching frozen reference is required")
        if config.test_type == "relative_loss":
            return PredictiveLossTest(
                reference,
                horizon=config.horizon,
                n_samples=config.n_samples,
                seed=config.seed,
            )
        from ..point_process_metric.ksd import KSD
        from .stein_test import SteinTest

        kernel = KSD(
            reference,
            horizon=config.horizon,
            bins=config.stein_bins,
            scale=config.stein_scale,
            sigma=config.stein_sigma,
            quadrature_order=config.quadrature_order,
            quadrature_rtol=config.quadrature_rtol,
            max_states=config.max_states,
        )
        return SteinTest(kernel, n_samples=config.n_samples, seed=config.seed)

    kernel = create_point_process_kernel(config)

    if config.test_type == "mmd":
        return MMDTwoSampleTest(kernel=kernel, n_samples=config.n_samples)
    else:
        raise ValueError(f"Unknown statistical test type: {config.test_type}")
