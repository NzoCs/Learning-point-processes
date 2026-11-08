"""Mathematical identities and public API tests, not a power-study substitute."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from new_ltpp.configs.window_test_config import WindowTestConfig
from new_ltpp.evaluation.statistical_testing.observations import (
    configurations,
    from_configurations,
)
from new_ltpp.evaluation.statistical_testing.point_process_metric.ksd import KSD
from new_ltpp.evaluation.statistical_testing.references import (
    HawkesReference,
    NeuralIntensityReference,
    PoissonReference,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    FixedWindowSignatureKernel,
    PredictiveLossTest,
    create_statistical_test,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.window_tests import (
    swap_signs,
)
from scripts.independent_simulator_validation import (
    constant_poisson_model,
    create_model,
)


def batch(rows):
    return from_configurations(
        [
            (
                torch.tensor(r, dtype=torch.float64),
                torch.zeros(len(r), dtype=torch.long),
            )
            for r in rows
        ]
    )


@pytest.mark.parametrize("representation", ["counting_grid", "intervals", "events"])
@pytest.mark.parametrize("space_kernel_type", ["rbf", "linear"])
def test_public_signature_full_gram_matches_explicit_pair_swaps(
    representation, space_kernel_type
):
    cfg = WindowTestConfig(
        test_type="window_mmd",
        horizon=1,
        num_event_types=1,
        representation=representation,
        space_kernel_type=space_kernel_type,
        grid=8,
        n_samples=7,
        seed=18,
    )
    test = create_statistical_test(cfg)
    x, y = batch([[], [0.2]]), batch([[0.3, 0.8], [0.4]])
    outcome = test.compute_statistics(x, y)
    pooled = batch([[], [0.2], [0.3, 0.8], [0.4]])
    gram = test.kernel.compute_gram_matrix(pooled, pooled)
    indices = torch.arange(2)
    expected = []
    for signs in swap_signs(2, 7, 18):
        swap = (signs < 0).long()
        expected.append(
            paired_joint_statistic(gram, indices + 2 * swap, indices + 2 * (1 - swap))
        )
    torch.testing.assert_close(outcome["permuted_statistics"], torch.stack(expected))
    assert outcome["p_value"] == test.compute_statistics(x, y)["p_value"]


def test_gap_path_retains_complete_and_censored_gaps_and_marks():
    rows = [
        (torch.tensor([0.2, 0.6], dtype=torch.float64), torch.tensor([1, 0])),
        (torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.long)),
    ]
    data = from_configurations(rows)
    kernel = FixedWindowSignatureKernel(representation="intervals", num_event_types=2)
    paths = kernel.paths(data)
    torch.testing.assert_close(
        0.02 * torch.expm1(paths[0, [3, 6, 9], 1]),
        torch.tensor([0.2, 0.4, 0.4], dtype=torch.float64),
    )
    assert paths[0, 3, 3] == 1 and paths[0, 6, 2] == 1
    assert paths[0, 9, 2:].sum() == 0
    assert torch.equal(paths[1, 3:], paths[1, 3:4].expand_as(paths[1, 3:]))


def test_censored_reflection_counterexample_loss_is_identical():
    ref = PoissonReference(1.0)

    # An independently specified reference with restricted mean .5 at origin
    # and cap afterwards yields identical relative squared errors at .4/.6.
    class Reference:
        num_event_types = 1

        def restricted_mean(self, t, m, origin, cap):
            return 0.5 if len(t) == 0 else cap

    test = PredictiveLossTest(Reference(), n_samples=15)
    x, y = batch([[0.4], [0.3]]), batch([[0.6], [0.7]])
    torch.testing.assert_close(test.losses(x), test.losses(y))
    assert test.compute_statistics(x, y)["p_value"] == 1
    torch.testing.assert_close(
        PredictiveLossTest(ref).losses(batch([[]])),
        torch.tensor(
            [((1 - (1 - math.exp(-1))) / (1 - math.exp(-1))) ** 2], dtype=torch.float64
        ),
    )


@pytest.mark.parametrize("kind", ["exp", "power"])
def test_papangelou_ratio_includes_future_events(kind):
    reference = HawkesReference(rate=3, beta=2, kind=kind, order=8)
    t, m = torch.tensor([0.2, 0.7], dtype=torch.float64), torch.tensor([0, 0])
    u = 0.4
    ratio = reference.papangelou(u, 0, t, m, 1)
    inserted = torch.tensor([0.2, 0.4, 0.7], dtype=torch.float64)
    direct = math.exp(
        reference.log_density(inserted, torch.tensor([0, 0, 0]), 1)
        - reference.log_density(t, m, 1)
    )
    assert ratio == pytest.approx(direct, rel=1e-10)
    causal = reference.intensities(t[:1], m[:1], 0.2, np.array([u - 0.2]))[0, 0]
    assert not np.isclose(ratio, causal)


def test_marked_stein_gram_is_exact_operator_sum_and_bootstrap_reproducible():
    ref = PoissonReference([1.0, 2.0])
    data = from_configurations(
        [
            (torch.tensor([0.2], dtype=torch.float64), torch.tensor([1])),
            (torch.empty(0), torch.empty(0, dtype=torch.long)),
        ]
    )
    kernel = KSD(ref, bins=1, scale=1, quadrature_order=2)
    states, weights = kernel.features(data)
    expected = torch.empty(2, 2, dtype=torch.float64)
    for i in range(2):
        for j in range(2):
            expected[i, j] = sum(
                weights[i, a]
                * weights[j, b]
                * torch.exp(-(states[i, a] - states[j, b]).square().sum() / 2)
                for a in range(5)
                for b in range(5)
            )
    torch.testing.assert_close(kernel.gram(data), expected)
    cfg = WindowTestConfig(
        test_type="ksd",
        horizon=1,
        num_event_types=2,
        stein_bins=1,
        quadrature_order=2,
        n_samples=19,
    )
    test = create_statistical_test(cfg, reference=ref)
    result = test.compute_statistics(data)
    assert result["null_method"] == "centered_multinomial_stein_bootstrap"
    torch.testing.assert_close(
        result["permuted_statistics"],
        test.compute_statistics(data)["permuted_statistics"],
    )
    with pytest.raises(ValueError, match="budget"):
        KSD(ref, max_states=1).gram(data)


@pytest.mark.parametrize("name", ["NHP", "THP", "SAHP", "RMTPP"])
def test_neural_reference_scores_empty_history_marks_and_terminal_exposure(
    name, tmp_path
):
    model, rates = constant_poisson_model(name, tmp_path)
    model = model.eval()
    # These architectures have no unconditional empty-history law in the public API.
    # A real, fixed observed history defines a conditional law without altering weights.
    history = (
        {} if name == "NHP" else dict(history_times=[0.0, 0.1], history_marks=[0, 1])
    )
    ref = NeuralIntensityReference(model, order=4, rtol=1e-5, **history)
    t, marks = torch.tensor([0.2, 0.7], dtype=torch.float64), torch.tensor([0, 1])
    expected = np.log(rates).sum() - rates.sum()
    assert ref.log_density(t, marks, 1) == pytest.approx(expected, rel=1e-5)
    assert ref.log_density(t[:0], marks[:0], 1) == pytest.approx(-rates.sum(), rel=1e-5)
    assert ref.papangelou(0.4, 1, t, marks, 1) == pytest.approx(rates[1], rel=1e-5)
    assert ref.restricted_mean(t, marks, 0.7, 0.3) == pytest.approx(
        -np.expm1(-rates.sum() * 0.3) / rates.sum(), rel=1e-5
    )


def test_existing_trained_nhp_weights_load_without_migration_and_are_not_mutated(
    tmp_path,
):
    model = create_model("NHP", tmp_path).double().eval()
    raw = json.loads(
        (Path(__file__).parents[1] / "integration/baselines/NHP.json").read_text()
    )
    state = {
        k: torch.tensor(v["values"], dtype=model.state_dict()[k].dtype).reshape(
            v["shape"]
        )
        for k, v in raw["trained_state"].items()
    }
    model.load_state_dict(state, strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ref = NeuralIntensityReference(model, order=8)
    value = ref.log_density(
        torch.tensor([0.2, 0.5], dtype=torch.float64), torch.tensor([0, 1]), 1
    )
    assert math.isfinite(value)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_invalid_exposure_padding_and_missing_reference_fail_explicitly():
    bad = batch([[0.0]])
    with pytest.raises(ValueError, match="BOS"):
        configurations(bad, 1, 1)
    with pytest.raises(ValueError, match="reference"):
        create_statistical_test(
            WindowTestConfig(test_type="ksd", horizon=1, num_event_types=1)
        )
    with pytest.raises(ValueError):
        WindowTestConfig(test_type="ksd", horizon=float("nan"), num_event_types=1)


def test_count_rbf_is_blind_to_timing_at_identical_counts():
    cfg = WindowTestConfig(
        test_type="window_mmd",
        horizon=1,
        num_event_types=1,
        point_process_kernel="count_rbf",
        sigma=1.0,
        n_samples=7,
    )
    result = create_statistical_test(cfg).compute_statistics(
        batch([[0.1], [0.2]]), batch([[0.8], [0.9]])
    )
    assert result["observed_statistic"] == 0 and result["p_value"] == 1
