"""Independent mathematical controls, with explicitly tracked current scientific defects."""

import numpy as np
import pytest
import torch

from scripts.independent_simulator_validation import (
    ALPHA,
    MU,
    POISSON_MODELS,
    constant_poisson_model,
    create_model,
    history,
)


@pytest.mark.parametrize("name", POISSON_MODELS)
def test_neural_constant_head_is_actual_constant_intensity(tmp_path, name):
    constant_poisson_model(name, tmp_path)


def test_self_correcting_post_event_intensity_matches_analytic_law(tmp_path):
    model = create_model(
        "SelfCorrecting", tmp_path, mu=MU.tolist(), alpha=ALPHA.tolist()
    )
    batch = history()
    actual = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(),
        sample_dtimes=torch.full((1, 3, 1), 0.4),
        compute_last_step_only=True,
    )
    expected = np.exp(
        MU * (float(batch.time_seqs[0, -1]) + 0.4)
        - ALPHA[:, batch.type_seqs[0].numpy()].sum(axis=1)
    )
    np.testing.assert_allclose(actual.detach().numpy().ravel(), expected, rtol=1e-6)


def test_self_correcting_small_growth_inverse_retains_nonzero_delay(
    tmp_path, monkeypatch
):
    model = create_model(
        "SelfCorrecting", tmp_path, mu=[1e-5, 1e-5], alpha=[[0.0, 0.0], [0.0, 0.0]]
    )
    monkeypatch.setattr(
        torch.Tensor,
        "exponential_",
        lambda tensor, *args, **kwargs: tensor.fill_(0.001),
    )
    actual = model.simulate(history(), num_events_to_simulate=1)
    expected = (
        np.log1p(0.001 * 1e-5 / np.exp(1e-5 * float(history().time_seqs[0, -1]))) / 1e-5
    )
    assert float(actual.time_delta_seqs[0, 0]) == pytest.approx(
        expected, rel=1e-5, abs=1e-8
    )


def test_fullynn_intensity_is_independent_of_unrelated_batch_rows():
    from new_ltpp.models.implementations.fullynn import CumulHazardFunctionNetwork

    torch.manual_seed(20261021)
    network = CumulHazardFunctionNetwork(hidden_size=4, num_event_types=2)
    _, single = network(torch.zeros(1, 1, 1, 4), torch.full((1, 1, 1), 0.5))
    _, batch = network(torch.zeros(4, 1, 1, 4), torch.full((4, 1, 1), 0.5))
    torch.testing.assert_close(single[0], batch[0], rtol=1e-6, atol=1e-9)


def test_independent_self_correcting_oracle_inverts_total_hazard():
    from scripts.independent_simulator_validation import self_correcting_oracle

    a, marks = self_correcting_oracle(np.random.default_rng(71), 16)
    b, _ = self_correcting_oracle(np.random.default_rng(71), 16)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and (a > 0).all()
    assert np.isin(marks, [0, 1]).all()
