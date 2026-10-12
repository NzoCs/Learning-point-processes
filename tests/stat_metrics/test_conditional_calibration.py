"""Exchangeability group, chronological holdout, and aggregation contracts."""

import itertools

import numpy as np
import pytest
import torch

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
    MMDTwoSampleTest,
    split_history_future,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import (
    cached_paired_statistics,
    conditional_hawkes_reference,
    conditional_poisson,
    exact_product_group_statistics,
    heterogeneous_histories,
)
from scripts.extended_scientific_validation import batch_from_arrays


def make_test(samples=19, conditional=True):
    cls = ConditionalMMDTwoSampleTest if conditional else MMDTwoSampleTest
    return cls(SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), samples)


def test_holdout_skips_padding_preserves_marks_and_resets_time_origin():
    batch = Batch(
        torch.tensor([[0.0, 0.2, 0.5, 1.0, 2.0], [0.1, 0.4, 0.9, 1.4, 0.0]]),
        torch.zeros(2, 5),
        torch.tensor([[2, 0, 1, 0, 1], [1, 0, 1, 0, 2]]),
        torch.tensor(
            [[False, True, True, True, True], [True, True, True, True, False]]
        ),
    )
    before = {k: v.clone() for k, v in batch.to_mapping().items()}
    h, f = split_history_future(batch, 2, 2)
    torch.testing.assert_close(h.time_seqs, torch.tensor([[0.2, 0.5], [0.1, 0.4]]))
    torch.testing.assert_close(f.time_seqs, torch.tensor([[0.5, 1.5], [0.5, 1.0]]))
    assert f.type_seqs.tolist() == [[0, 1], [1, 0]]
    torch.testing.assert_close(
        f.time_delta_seqs, torch.tensor([[0.5, 1.0], [0.5, 0.5]])
    )
    for k, v in before.items():
        torch.testing.assert_close(v, batch.to_mapping()[k])
    with pytest.raises(ValueError, match="Not enough"):
        split_history_future(batch, 3, 2)
    with pytest.raises(ValueError, match="positive"):
        split_history_future(batch, 0, 2)


def test_all_pair_swaps_match_public_mmd_and_commute_with_truncation(monkeypatch):
    rng = np.random.default_rng(8)
    h = heterogeneous_histories(3)
    x, y = conditional_poisson(rng, h), conditional_poisson(rng, h)
    rows = torch.arange(3)
    orders = torch.stack(
        [
            torch.cat((rows + 3 * torch.tensor(b), rows + 3 * (1 - torch.tensor(b))))
            for b in itertools.product((0, 1), repeat=3)
        ]
    )
    test = make_test(len(orders) - 1)
    stats = cached_paired_statistics(test, h, x, y, orders)
    pool = test._concat_batches(*test._truncate_to_min_time(x, y))
    future = test.kernel.compute_gram_matrix(pool, pool)
    history = test.kernel.compute_gram_matrix(h, h)
    weights = torch.exp(
        -0.5
        * (history.diag()[:, None] + history.diag()[None, :] - 2 * history).clamp_min(0)
    )
    for order, stat in zip(orders, stats):
        assert torch.equal(order[:3] % 3, rows) and torch.equal(order[3:] % 3, rows)
        ix, iy = order[:3], order[3:]
        # Literal scalar oracle, independent of the production block summation.
        expected = (
            sum(
                weights[i, j]
                * (
                    future[ix[i], ix[j]]
                    + future[iy[i], iy[j]]
                    - future[ix[i], iy[j]]
                    - future[iy[i], ix[j]]
                )
                for i in range(3)
                for j in range(3)
                if i != j
            )
            / 6
        )
        torch.testing.assert_close(stat, expected)
    iterator = iter(orders[1:])
    import new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test as module

    def draw(*args):
        order = next(iterator)
        return order[:3], order[3:]

    monkeypatch.setattr(module, "pair_swap_indices", draw)
    actual = test.compute_statistics(x, y, histories=h)
    torch.testing.assert_close(actual["observed_statistic"], stats[0])
    torch.testing.assert_close(actual["permuted_statistics"], stats[1:])
    assert actual["null_method"] == "within_history_pair_swaps_joint_kernel_v1"


def test_exact_product_group_controls_type_I_for_every_orientation():
    rng = np.random.default_rng(42)
    h = heterogeneous_histories(3)
    batches = [
        (h, conditional_poisson(rng, h), conditional_poisson(rng, h)) for _ in range(2)
    ]
    values = exact_product_group_statistics(make_test(), batches)
    assert values.numel() == 64
    p = (
        (
            (values[:, None] >= values[None, :])
            | torch.isclose(values[:, None], values[None, :], rtol=1e-10, atol=1e-12)
        )
        .double()
        .mean(0)
    )
    for alpha in (0.05, 0.1, 0.2, 0.5):
        assert float((p <= alpha).double().mean()) <= alpha + 1e-12


def test_unequal_batches_sum_null_columns_and_reset_without_averaging_p_values():
    test = make_test()
    rng = np.random.default_rng(55)
    observed, null = [], []
    for i, n in enumerate((5, 3)):
        h = heterogeneous_histories(n)
        x, y = conditional_poisson(rng, h), conditional_poisson(rng, h)
        torch.manual_seed(i)
        result = test.compute_statistics(x, y, histories=h)
        observed.append(result["observed_statistic"])
        null.append(result["permuted_statistics"])
    torch.testing.assert_close(
        test.get_final_p_value(), monte_carlo_p_value(sum(observed), sum(null))
    )
    assert test.n_batches == 2
    with pytest.raises(ValueError, match="counts"):
        test._accumulate(torch.tensor(1.0), torch.zeros(3))
    test.reset_accumulators()
    assert test.n_batches == 0 and test.get_final_p_value() == 1


@pytest.mark.parametrize(
    "observed,null", [(float("nan"), [0.0]), (0.0, [float("inf")]), (0.0, [])]
)
def test_nonfinite_or_empty_null_cannot_produce_a_significant_p_value(observed, null):
    with pytest.raises(ValueError):
        monte_carlo_p_value(torch.tensor(observed), torch.tensor(null))


def test_conditional_api_requires_histories_and_matching_pairs():
    h = heterogeneous_histories(3)
    x = conditional_poisson(np.random.default_rng(8), h)
    test = make_test()
    with pytest.raises(ValueError, match="shared histories"):
        test.compute_statistics(x, x)
    with pytest.raises(ValueError, match="matched"):
        test.compute_statistics(x, x, histories=heterogeneous_histories(2))
    with pytest.raises(ValueError, match="explicit histories"):
        test.test_simulation([x], [x])
    with pytest.raises(ValueError, match="history_events"):
        make_test(conditional=False).test_model(None, [], None)
    bad = batch_from_arrays(x.time_seqs.numpy() + 1, x.type_seqs.numpy())
    bad.time_delta_seqs = x.time_delta_seqs.clone()
    with pytest.raises(ValueError, match="origin"):
        test.compute_statistics(x, bad, histories=h)


def test_model_api_conditions_on_prefix_only_and_restarts_aggregation():
    h = heterogeneous_histories(3)
    future = conditional_poisson(np.random.default_rng(8), h, events=2)
    full = batch_from_arrays(
        torch.cat(
            (h.time_seqs.double(), future.time_seqs + h.time_seqs[:, -1:].double()),
            dim=1,
        ).numpy(),
        torch.cat((h.type_seqs, future.type_seqs), dim=1).numpy(),
    )

    class Model:
        training = False
        _simulator = None
        calls = 0

        def compute_intensities_at_sample_dtimes(self, **kwargs):
            raise AssertionError("Calibration must call the model's native simulator")

        def simulate(self, prefix, num_events_to_simulate):
            assert num_events_to_simulate == 2
            torch.testing.assert_close(prefix.time_seqs, h.time_seqs.double())
            self.calls += 1
            return split_history_future(full, 3, 2)[1]

    model = Model()
    test = make_test(3)
    first = test.test_model(model, [full], None, history_events=3, future_events=2)
    assert first["p_value"] == 1 and test.n_batches == 1
    second = test.test_model(model, [full], None, history_events=3, future_events=2)
    assert second == first and test.n_batches == 1 and model.calls == 2
    delegated = make_test(3, conditional=False)
    result = delegated.test_model(
        model, [full], None, history_events=3, future_events=2
    )
    assert result == first
    assert (
        delegated.n_batches == 1
        and float(delegated.get_final_p_value()) == result["p_value"]
    )
    with pytest.raises(ValueError, match="null schemes"):
        delegated.compute_statistics(future, future)
    model.training = True
    with pytest.raises(ValueError, match="frozen"):
        test.test_model(model, [full], None, history_events=3, future_events=2)


def test_unconditional_loader_call_resets_previous_run_and_rejects_missing_batches():
    h = heterogeneous_histories(3)
    x = conditional_poisson(np.random.default_rng(8), h)
    test = make_test(3, conditional=False)
    test._accumulate(torch.tensor(100.0), torch.zeros(3))
    torch.manual_seed(1)
    first = test.test_simulation([x], [x])
    assert test.n_batches == 1
    torch.manual_seed(1)
    second = test.test_simulation([x], [x])
    assert first == second and test.n_batches == 1
    with pytest.raises(ValueError, match="same number"):
        test.test_simulation([x, x], [x])
    assert test.n_batches == 0


def test_oracle_initializes_excitation_from_actual_history():
    class FixedRng:
        scales = []

        def exponential(self, scale):
            self.scales.append(scale)
            return 1.0

        def random(self):
            return 0.0

        def choice(self, k, p):
            return 0

    rng = FixedRng()
    h = batch_from_arrays([[0.2, 0.5]], [[1, 0]])
    conditional_hawkes_reference(
        rng, h, [1.0, 1.0], [[0.2, 0.3], [0.1, 0.4]], [[1.0, 2.0], [1.0, 2.0]], events=1
    )
    expected = 2.0 + 0.2 + 0.1 + (0.3 * 2 + 0.4 * 2) * np.exp(-2 * 0.3)
    assert rng.scales[0] == pytest.approx(1 / expected)


def test_ragged_hawkes_oracle_ignores_left_and_right_padding():
    h = heterogeneous_histories(2, ragged=True)
    alternative = Batch(*(v.clone() for v in h.to_mapping().values()))
    alternative.time_seqs[1] = torch.tensor([0.2, 5.0, 0.0, 0.0, 0.0, 0.0])
    alternative.type_seqs[1] = torch.tensor([1, 1, 2, 2, 2, 2])
    alternative.valid_event_mask[1] = torch.tensor(
        [True, True, False, False, False, False]
    )
    parameters = dict(
        mu=[1.0, 1.0], alpha=[[0.2, 0.3], [0.1, 0.4]], beta=[[1.0, 2.0], [1.0, 2.0]]
    )
    a = conditional_hawkes_reference(np.random.default_rng(88), h, **parameters)
    b = conditional_hawkes_reference(
        np.random.default_rng(88), alternative, **parameters
    )
    for key, value in a.to_mapping().items():
        torch.testing.assert_close(value, b.to_mapping()[key], rtol=0, atol=0)


def test_conditional_accumulator_preserves_trajectory_rng_and_uses_product_sum():
    from new_ltpp.configs import StatisticalTestConfig
    from new_ltpp.evaluation.accumulators.statistical_metrics_accumulator import (
        StatisticalTestAccumulator,
    )

    config = StatisticalTestConfig(
        test_type="mmd",
        point_process_kernel_type="sig_kernel",
        space_kernel_type="rbf",
        num_event_types=2,
        n_samples=19,
        num_discretization_points=8,
        dyadic_order=0,
        embedding_type="counting_grid",
    )
    acc = StatisticalTestAccumulator(config, conditional=True)
    h = heterogeneous_histories(3)
    rng = np.random.default_rng(8)
    x, y = conditional_poisson(rng, h), conditional_poisson(rng, h)
    before = torch.get_rng_state().clone()
    acc.update(x, y, histories=h)
    state = acc._null_rng_states["cpu"].clone()
    acc.update(x, y, histories=h)
    assert torch.equal(before, torch.get_rng_state()) and not torch.equal(
        state, acc._null_rng_states["cpu"]
    )
    assert (
        acc.compute()["null_methods"]
        == ["within_history_pair_swaps_joint_kernel_v1"] * 2
    )
    assert acc.compute()["pooled_p_value"] == float(acc.test.get_final_p_value())
    with pytest.raises(ValueError, match="mode"):
        acc.update(x, y)
    acc.reset()
    assert acc.test.n_batches == 0 and not acc._null_rng_states


def test_pair_identifiers_reject_overlapping_and_unchecked_batches():
    h = heterogeneous_histories(3)
    rng = np.random.default_rng(5)
    x, y = conditional_poisson(rng, h), conditional_poisson(rng, h)
    test = make_test(3)
    test.compute_statistics(x, y, histories=h, pair_ids=[0, 1, 2])
    with pytest.raises(ValueError, match="Overlapping"):
        test.compute_statistics(x, y, histories=h, pair_ids=[2, 3, 4])
    assert test.n_batches == 1
    with pytest.raises(ValueError, match="All accumulated"):
        test.compute_statistics(x, y, histories=h)
    with pytest.raises(ValueError, match="distinct"):
        test.compute_statistics(x, y, histories=h, pair_ids=[3, 3, 4])
    test.compute_statistics(x, y, histories=h, pair_ids=[3, 4, 5])
    assert test.n_batches == 2
    test.reset_accumulators()
    test.compute_statistics(x, y, histories=h)
    with pytest.raises(ValueError, match="All accumulated"):
        test.compute_statistics(x, y, histories=h, pair_ids=[0, 1, 2])
