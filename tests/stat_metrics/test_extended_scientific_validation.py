"""Independent oracle properties and real fitted-model checkpoint replay."""

import numpy as np
import pytest
import torch

from scripts.extended_scientific_validation import (
    batch_from_arrays,
    fit_model,
    marked_reference,
    model_pool,
)


def test_marked_poisson_has_correct_gaps_and_mark_probabilities():
    batch = marked_reference(np.random.default_rng(45), "poisson", 1000, 3)
    assert abs(float(batch.time_delta_seqs.mean()) - 0.5) < 0.03
    assert abs(float((batch.type_seqs == 0).double().mean()) - 0.65) < 0.03
    assert batch.valid_event_mask.all()
    assert (batch.time_delta_seqs > 0).all()


def test_zero_excitation_hawkes_is_exactly_the_poisson_oracle():
    kwargs = dict(size=5, events=4, alpha=np.zeros((2, 2)))
    a = marked_reference(np.random.default_rng(14), "poisson", **kwargs)
    b = marked_reference(np.random.default_rng(14), "hawkes", **kwargs)
    for key, value in a.to_mapping().items():
        torch.testing.assert_close(value, b.to_mapping()[key], rtol=0, atol=0)


def test_mark_probability_uses_post_decay_intensity():
    class RecordedRng:
        probabilities = []

        def exponential(self, scale):
            return 1.0

        def random(self):
            return 0.0

        def choice(self, k, p):
            self.probabilities.append(p.copy())
            return 0

    rng = RecordedRng()
    batch = marked_reference(
        rng,
        "hawkes",
        1,
        2,
        mu=[1.0, 1.0],
        alpha=[[0.5, 0.0], [0.8, 0.0]],
        beta=[[1.0, 1.0], [2.0, 1.0]],
    )
    expected = np.array([1 + 0.5 * np.exp(-1), 1 + 1.6 * np.exp(-2)])
    np.testing.assert_allclose(rng.probabilities[1], expected / expected.sum())
    torch.testing.assert_close(
        batch.time_seqs, torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    )


@pytest.mark.parametrize("name", ["Hawkes", "NHP"])
def test_fitted_model_checkpoint_replays_simulations(name, tmp_path):
    torch.set_num_threads(1)
    train = marked_reference(np.random.default_rng(18), "hawkes", 8, 4)
    model, metadata = fit_model(name, train, tmp_path, seed=29, steps=2)
    assert metadata["steps"] == 2
    a = model_pool(model, 4, 3, seed=30)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    model.load_state_dict(torch.load(tmp_path / f"{name}.pt", weights_only=True))
    b = model_pool(model, 4, 3, seed=30)
    for key, tensor in a.to_mapping().items():
        torch.testing.assert_close(tensor, b.to_mapping()[key], rtol=0, atol=0)
    assert ((a.type_seqs >= 0) & (a.type_seqs < 2)).all()
    assert (a.time_delta_seqs > 0).all()


def test_batch_preserves_first_waiting_time():
    batch = batch_from_arrays([[0.5, 1.2]], [[1, 0]])
    torch.testing.assert_close(
        batch.time_delta_seqs, torch.tensor([[0.5, 0.7]], dtype=torch.float64)
    )


def test_marked_experiment_matches_public_permutation_statistics(monkeypatch):
    from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
    from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
        RBFKernel,
    )
    from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
        MMDTwoSampleTest,
    )
    from scripts.compare_truncation import paired_statistics

    rng = np.random.default_rng(71)
    x = marked_reference(rng, "hawkes", 2)
    y = marked_reference(rng, "hawkes", 2)
    test = MMDTwoSampleTest(SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), 3)
    orders = torch.stack(
        [
            torch.arange(4),
            torch.tensor([2, 0, 1, 3]),
            torch.tensor([3, 1, 0, 2]),
            torch.tensor([1, 3, 2, 0]),
        ]
    )
    expected = paired_statistics(test, x, y, orders)["recomputed_pairwise_truncation"]
    iterator = iter(orders[1:])
    monkeypatch.setattr(torch, "randperm", lambda *a, **kw: next(iterator))
    actual = test.compute_statistics(x, y, accumulate=False)
    torch.testing.assert_close(expected[0], actual["observed_statistic"])
    torch.testing.assert_close(expected[1:], actual["permuted_statistics"])
