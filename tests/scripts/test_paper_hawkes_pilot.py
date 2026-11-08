import numpy as np
import pytest
import torch
from scipy.integrate import quad

from new_ltpp.data.generation.convergent_hawkes import kernel, simulate, tail_mass
from new_ltpp.evaluation.statistical_testing.observations import from_configurations
from new_ltpp.evaluation.statistical_testing.statistical_tests.window_tests import (
    FixedWindowSignatureKernel,
)
from scripts.paper_hawkes_pilot import algebra


@pytest.mark.parametrize("p", [None, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0])
def test_kernel_mass_tail_and_convergent_amplitude(p):
    value, error = quad(lambda t: kernel(t, p=p), 0, np.inf)
    assert value == pytest.approx(0.5, abs=1e-9)
    assert kernel(0, p=p) == 5
    finite, error = quad(lambda t: kernel(t, p=p), 0, 10)
    assert finite + tail_mass(10, p=p) == pytest.approx(0.5, abs=1e-8)


def test_convergence_and_cluster_exposure_retains_ancestors():
    t = np.array([0.0, 0.01, 0.1, 0.5])
    np.testing.assert_allclose(kernel(t, p=100000), kernel(t), rtol=0.001, atol=1e-6)
    visible, full = simulate(np.random.default_rng(9), burn=10, p=3)
    assert (full < 0).any() and np.all(np.diff(full) > 0)
    np.testing.assert_array_equal(visible, full[(full > 0) & (full < 1)])
    with pytest.raises(RuntimeError, match="budget"):
        simulate(np.random.default_rng(9), burn=10, max_events=1)


def test_grid_collision_and_event_path_differ_on_same_trajectories():
    data = from_configurations(
        [
            (torch.tensor(v, dtype=torch.float64), torch.zeros(2, dtype=torch.long))
            for v in [[0.1, 0.11], [0.1, 0.49]]
        ]
    )
    grid = FixedWindowSignatureKernel(grid=3).paths(data)
    event = FixedWindowSignatureKernel(representation="events").paths(data)
    assert torch.equal(grid[0], grid[1])
    assert not torch.equal(event[0], event[1])
    assert event[0, -1, 1] == 1 and event[0, -1, 0] == 0.2
    assert algebra()["population_mmd_squared"] == 0
