"""Independent laws and pointwise differentiation for defective waiting times."""

import math

import pytest
import torch

from new_ltpp.models.hazard_sampling import invert_hazard
from new_ltpp.models.implementations.fullynn import CumulHazardFunctionNetwork


def test_fullynn_derivatives_are_pointwise_and_independent_of_batch_and_grid():
    torch.manual_seed(62)
    net = CumulHazardFunctionNetwork(hidden_size=4, num_event_types=2).double()
    hidden = torch.zeros(1, 1, 1, 4, dtype=torch.double)
    t = torch.full((1, 1, 1), 0.2, dtype=torch.double)
    _, reference = net(hidden, t.clone())
    for shape in ((4, 3, 7), (1, 1, 1)):
        _, derivative = net(hidden.expand(*shape, 4), t.expand(shape).clone())
        torch.testing.assert_close(derivative, reference.expand(*shape, 2))
    step = 1e-5
    plus, _ = net(hidden, t + step)
    minus, _ = net(hidden, t - step)
    torch.testing.assert_close(
        reference, (plus - minus) / (2 * step), rtol=1e-6, atol=1e-9
    )


def test_inverse_hazard_has_correct_atom_and_finite_time_distribution():
    # lambda(t)=2 exp(-t), H(t)=2(1-exp(-t)). The atom is exp(-2).
    torch.manual_seed(180)
    limit = torch.tensor([[2.0]], dtype=torch.double)
    hazard = lambda t: -2 * torch.expm1(-t)
    draws = invert_hazard(hazard, limit, 8192)
    atom = torch.isinf(draws).double().mean().item()
    assert abs(atom - math.exp(-2)) < 0.015
    finite = draws[torch.isfinite(draws)]
    for t in (0.2, 0.7, 2.0):
        # Unconditional finite-time CDF, with the atom retained in denominator.
        assert (
            abs(
                (draws <= t).double().mean().item()
                - (1 - math.exp(-2 * (1 - math.exp(-t))))
            )
            < 0.02
        )
    assert finite.numel() > 6000
    torch.manual_seed(180)
    torch.testing.assert_close(
        draws, invert_hazard(hazard, limit, 8192), rtol=0, atol=0
    )
    conditional = invert_hazard(hazard, limit, 8192, conditional=True)
    assert torch.isfinite(conditional).all()
    expected = (1 - math.exp(-2 * (1 - math.exp(-0.7)))) / (1 - math.exp(-2))
    assert abs((conditional <= 0.7).double().mean().item() - expected) < 0.02


def test_zero_hazard_is_absence_not_a_budget_failure():
    samples = invert_hazard(lambda t: torch.zeros_like(t), torch.zeros(2, 1), 3)
    assert torch.isinf(samples).all()


def test_conditional_sampler_does_not_round_rare_draws_to_infinity():
    from unittest.mock import patch

    def large_threshold(tensor, *args, **kwargs):
        return tensor.fill_(30.0)

    with patch.object(torch.Tensor, "exponential_", large_threshold):
        values = invert_hazard(
            lambda t: t, torch.tensor([[100.0]]), 1, conditional=True
        )
    torch.testing.assert_close(values, torch.tensor([[[30.0]]]))


def test_fullynn_last_history_and_compensator_match_intensity(tmp_path):
    from scripts.independent_simulator_validation import create_model, history

    model = create_model("FullyNN", tmp_path)
    batch = history(2)
    queries = torch.full((2, 3, 5), 0.2)
    full = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(), sample_dtimes=queries
    )
    last = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(), sample_dtimes=queries[:, -1:], compute_last_step_only=True
    )
    torch.testing.assert_close(last, full[:, -1:])
    hazard, limit = model.next_event_hazard(
        **batch.to_mapping(), compute_last_step_only=True
    )
    assert (limit >= 0).all() and torch.isfinite(limit).all()
    torch.testing.assert_close(hazard(torch.zeros(2, 1, 1)), torch.zeros(2, 1, 1))
    times = torch.linspace(0, 2, 1001).expand(2, 1, -1).clone()
    intensities = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(), sample_dtimes=times, compute_last_step_only=True
    ).sum(-1)
    integral = torch.trapezoid(intensities, times, dim=-1)
    torch.testing.assert_close(
        integral, hazard(torch.full((2, 1, 1), 2.0)).squeeze(-1), rtol=1e-3, atol=1e-6
    )


def test_hazard_bracket_failure_is_not_misreported_as_absence():
    with pytest.raises(RuntimeError, match="bracket"):
        invert_hazard(
            lambda t: torch.zeros_like(t),
            torch.full((1, 1), torch.inf),
            1,
            max_rounds=3,
        )


def test_invalid_hazard_cannot_be_returned_as_a_finite_event():
    with pytest.raises(RuntimeError, match="Invalid integrated intensity"):
        invert_hazard(
            lambda t: torch.full_like(t, torch.nan), torch.full((1, 1), torch.inf), 1
        )


@pytest.mark.parametrize(
    "slope,log_rate", [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (-1.0, 12.0), (1.0, 12.0)]
)
def test_rmtpp_compensator_matches_independent_quadrature(tmp_path, slope, log_rate):
    from scripts.independent_simulator_validation import constant_poisson_model, history

    model, _ = constant_poisson_model("RMTPP", tmp_path)
    with torch.no_grad():
        model.w_t.fill_(slope)
        model.b_t.fill_(log_rate)
    batch = history()
    hazard, limit = model.next_event_hazard(
        **batch.to_mapping(), compute_last_step_only=True
    )
    t = torch.linspace(0, 3, 20001, dtype=torch.double)
    expected = torch.trapezoid(
        2 * (log_rate + slope * t).clamp(max=math.log(1e5)).exp(), t
    )
    torch.testing.assert_close(
        hazard(torch.tensor([[[3.0]]], dtype=torch.double)).squeeze(),
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    assert bool(torch.isfinite(limit).all()) == (slope < 0)


def test_simulation_stops_only_terminated_rows_and_saves_empty_sequences(tmp_path):
    import json

    from new_ltpp.models.simulation.simulator import Simulator
    from new_ltpp.models.simulation.tpp_io import SimulationIOManager
    from scripts.independent_simulator_validation import history

    class Model:
        device = torch.device("cpu")
        pad_token_id = 2
        num_event_types = 2
        compute_intensities_at_sample_dtimes = None

        def sample_next_event(self, *, type_seqs, **kwargs):
            stopped = type_seqs[:, 0] == 1
            dt = torch.where(stopped, torch.inf, 0.5).unsqueeze(-1)
            return dt, torch.zeros_like(dt, dtype=torch.long)

    batch = history(2)
    batch.type_seqs[0, 0] = 1
    result = Simulator(Model()).simulate(batch, 3)
    assert result.valid_event_mask.tolist() == [[False] * 3, [True] * 3]
    assert result.no_next_event_mask.tolist() == [True, False]
    assert result.time_delta_seqs[1].tolist() == [0.5] * 3
    assert torch.isfinite(result.time_seqs).all()
    io = SimulationIOManager(2)
    io.setup_io(tmp_path)
    io.update(result)
    io.finalize()
    statuses = [
        json.loads(line)
        for line in (tmp_path / "sequence_status.jsonl").read_text().splitlines()
    ]
    assert statuses[0]["no_next_event"] and statuses[0]["generated_events"] == 0
    assert not statuses[1]["no_next_event"]


def test_entirely_absent_futures_remain_in_statistics_and_empty_parquet(tmp_path):
    import pyarrow.parquet as pq

    from new_ltpp.evaluation.accumulators.event_type_accumulator import (
        EventTypeAccumulator,
    )
    from new_ltpp.evaluation.accumulators.len_accumulator import (
        SequenceLengthAccumulator,
    )
    from new_ltpp.evaluation.accumulators.time_accumulator import (
        InterEventTimeAccumulator,
    )
    from new_ltpp.models.simulation.tpp_io import SimulationIOManager
    from scripts.independent_simulator_validation import history

    batch = history(2)
    empty = history(2)
    empty.valid_event_mask.zero_()
    empty.no_next_event_mask = torch.ones(2, dtype=torch.bool)
    lengths = SequenceLengthAccumulator()
    lengths.update(batch, empty)
    result = lengths.compute()
    assert result["gt_count"] == result["sim_count"] == 2
    assert result["sim_mean"] == 0
    marks = EventTypeAccumulator(2)
    marks.update(batch, empty)
    assert marks.compute()["gt_count"] == 6
    times = InterEventTimeAccumulator(0, 2)
    times.update(batch, empty)
    assert times.compute()["gt_count"] == 6
    io = SimulationIOManager(2)
    io.setup_io(tmp_path)
    io.update(empty)
    io.finalize()
    assert pq.read_table(tmp_path / "sim_batch.parquet").num_rows == 0
