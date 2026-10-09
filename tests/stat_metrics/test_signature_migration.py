"""Real pySigLib checks against the pinned legacy finite-difference recurrence.

The recurrence oracle is independent Python math, not the legacy native library.
Reference: sigkernel 40a5831, sigkernel/cython_backend.pyx, non-naive Gram solver.
"""

import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels.signature_backend import (
    PySigLibKernel,
    unbiased_mmd_squared,
)
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    LinearKernel,
    RBFKernel,
)


def legacy_recurrence(x, y, kernel, dyadic_order):
    spatial = kernel.Gram_matrix(x, y)
    increments = (
        spatial[..., 1:, 1:]
        + spatial[..., :-1, :-1]
        - spatial[..., 1:, :-1]
        - spatial[..., :-1, 1:]
    )
    refinement = 2**dyadic_order
    increments = increments.repeat_interleave(refinement, -2).repeat_interleave(
        refinement, -1
    )
    increments = increments / refinement**2
    n, m, height, width = increments.shape
    grid = torch.ones((n, m, height + 1, width + 1), dtype=x.dtype, device=x.device)
    for row in range(height):
        for col in range(width):
            q = increments[..., row, col]
            correction = q.square() / 12
            grid[..., row + 1, col + 1] = (
                grid[..., row, col + 1] + grid[..., row + 1, col]
            ) * (1 + q / 2 + correction) - grid[..., row, col] * (1 - correction)
    return grid[..., -1, -1]


@pytest.fixture
def paths():
    generator = torch.Generator().manual_seed(711)
    return (
        torch.randn((3, 4, 3), dtype=torch.float64, generator=generator) * 0.1,
        torch.randn((2, 5, 3), dtype=torch.float64, generator=generator) * 0.1,
    )


@pytest.mark.parametrize(
    "kernel", [LinearKernel(), LinearKernel(2.5), RBFKernel(0.3), RBFKernel(0.3, 2.5)]
)
@pytest.mark.parametrize("order", [0, 1, 2])
def test_matches_pinned_solver_recurrence(paths, kernel, order):
    x, y = paths
    expected = legacy_recurrence(x, y, kernel, order)
    actual = PySigLibKernel(kernel, order, max_batch=2).compute_Gram(x, y)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kernel", [LinearKernel(2.5), RBFKernel(0.3, 2.5)])
def test_batch_limit_does_not_change_gram(paths, kernel):
    x, y = paths
    small = PySigLibKernel(kernel, 1, max_batch=1).compute_Gram(x, y)
    large = PySigLibKernel(kernel, 1, max_batch=64).compute_Gram(x, y)
    torch.testing.assert_close(small, large, rtol=1e-10, atol=1e-10)


def test_preserves_unbiased_mmd_with_unequal_sample_sizes(paths):
    x, y = paths
    static = RBFKernel(0.3, 2.5)
    kxx = legacy_recurrence(x, x, static, 1)
    kxy = legacy_recurrence(x, y, static, 1)
    kyy = legacy_recurrence(y, y, static, 1)
    expected = (
        (kxx.sum() - kxx.trace()) / 6 + (kyy.sum() - kyy.trace()) / 2 - 2 * kxy.mean()
    )
    actual = PySigLibKernel(static, 1).compute_mmd(x, y)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_unbiased_mmd_can_be_negative():
    identity = torch.eye(2, dtype=torch.float64)
    assert unbiased_mmd_squared(identity, identity, identity).item() == -1.0


def test_rejects_single_sample_mmd(paths):
    x, y = paths
    with pytest.raises(ValueError, match="at least two"):
        PySigLibKernel(LinearKernel(), 0).compute_mmd(x[:1], y)


@pytest.mark.parametrize("kernel", [LinearKernel(2.5), RBFKernel(0.3, 2.5)])
def test_gradients_follow_solver_finite_differences(kernel):
    generator = torch.Generator().manual_seed(91)
    x = (
        torch.randn((1, 3, 2), generator=generator, dtype=torch.float64) * 0.1
    ).requires_grad_()
    y = (
        torch.randn((1, 3, 2), generator=generator, dtype=torch.float64) * 0.1
    ).requires_grad_()
    backend = PySigLibKernel(kernel, 1)
    assert torch.autograd.gradcheck(
        backend.compute_Gram, (x, y), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_legacy_backend_configuration_is_rejected():
    from pydantic import ValidationError

    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig

    with pytest.raises(ValidationError, match="signature_backend"):
        StatisticalTestConfig(
            test_type="mmd",
            point_process_kernel_type="sig_kernel",
            space_kernel_type="linear",
            num_event_types=2,
            n_samples=10,
            signature_backend="sigkernel",
        )


def test_project_embedding_with_padding_and_empty_sequence():
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.sig_kernel import (
        SIGKernel,
    )
    from new_ltpp.shared_types import Batch

    times = torch.tensor([[0.1, 0.5, 0.9], [0.2, 8.0, 8.0], [8.0, 8.0, 8.0]])
    mask = torch.tensor(
        [[True, True, True], [True, False, False], [False, False, False]]
    )
    sample = Batch(
        times, torch.zeros_like(times), torch.zeros_like(times, dtype=torch.long), mask
    )
    other = Batch(
        times[:2],
        torch.zeros_like(times[:2]),
        torch.zeros_like(times[:2], dtype=torch.long),
        mask[:2],
    )
    static = RBFKernel(0.3, 2.5)
    kernel = SIGKernel(static, "linear", 8, 1, 1)
    x, y = kernel._prepare_kernel(sample, other)
    expected = legacy_recurrence(x, y, static, 1)
    actual = kernel.compute_gram_matrix(sample, other)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    assert torch.isfinite(actual).all()


def test_factory_uses_configured_signature_backend():
    from new_ltpp.configs.statistical_test_config import StatisticalTestConfig
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.factory import (
        create_point_process_kernel,
    )

    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="linear",
        num_event_types=2,
        n_samples=10,
        signature_max_batch=2,
    )
    kernel = create_point_process_kernel(config)
    assert kernel.backend_name == "pysiglib"
    assert kernel.kernel.max_batch == 2


@pytest.mark.parametrize("order", [-1, 0.5, True])
def test_invalid_refinement_is_rejected(order):
    with pytest.raises(ValueError, match="dyadic_order"):
        PySigLibKernel(LinearKernel(), order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cuda_matches_cpu_without_cpu_fallback(paths):
    x, y = paths
    backend = PySigLibKernel(RBFKernel(0.3, 2.5), 1)
    cpu = backend.compute_Gram(x, y)
    gpu = backend.compute_Gram(x.cuda(), y.cuda())
    assert gpu.is_cuda
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-8, atol=1e-8)
