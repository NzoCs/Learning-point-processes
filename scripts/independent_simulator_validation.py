"""External simulator references and explicit diagnostics; does not change model calculations."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import kstest

from new_ltpp.configs import ModelConfig
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.shared_types import Batch
from scripts.extended_scientific_validation import code_provenance
from scripts.integration_suite import FIXTURES, write_json

POISSON_MODELS = ("ANHN", "ANHP", "NHP", "ODETPP", "RMTPP", "SAHP", "THP")
MU = np.array([0.15, 0.3])
ALPHA = np.array([[0.2, 0.4], [0.1, 0.25]])


def history(size=1):
    times = torch.tensor([[0.1, 0.3, 0.8]]).repeat(size, 1)
    return Batch(
        times,
        torch.diff(times, prepend=torch.zeros(size, 1)),
        torch.tensor([[0, 1, 0]]).repeat(size, 1),
        torch.ones(size, 3, dtype=torch.bool),
    )


def create_model(name, output, initial_window=20.0, **kwargs):
    import new_ltpp.models  # noqa: F401 -- public registry

    torch.manual_seed(20261015)
    raw = json.loads((FIXTURES / "config.json").read_text())["model_config"]
    raw["model_id"] = name
    raw["thinning_config"].update(
        num_sample=1, num_exp=32, over_sample_rate=5.0, num_samples_boundary=16
    )
    model = ModelFactory.create_model_by_name(
        name,
        ModelConfig.model_validate(raw),
        dict(
            num_event_types=2,
            pad_token_id=2,
            end_time_max=100.0,
            dtime_max=initial_window,
        ),
        output / name,
        **kwargs,
    )
    model.eval()
    model._simulator = Simulator(model)
    return model


def zero_linear(module):
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()


def constant_poisson_model(name, output, initial_window=20.0):
    model = create_model(name, output, initial_window)
    with torch.no_grad():
        if name == "RMTPP":
            zero_linear(model.hidden_to_intensity_logits)
            model.b_t.zero_()
            model.w_t.zero_()
            rates = np.ones(2)
        elif name in ("THP", "SAHP"):
            zero_linear(model.layer_intensity_hidden)
            model.softplus.log_beta.zero_()
            if name == "SAHP":
                zero_linear(model.mu)
                zero_linear(model.eta)
            if name == "THP":
                model.factor_intensity_base.zero_()
                model.factor_intensity_decay.zero_()
            rates = np.full(2, math.log(2))
        else:
            zero_linear(model.layer_intensity)
            if name == "ANHP":
                model.softplus.log_beta.zero_()
            rates = np.full(2, math.log(2))
    batch = history(2)
    samples = torch.linspace(0, 20, 17).reshape(1, 1, 17).repeat(2, 3, 1)
    actual = model.compute_intensities_at_sample_dtimes(
        **batch.to_mapping(), sample_dtimes=samples, compute_last_step_only=True
    )
    torch.testing.assert_close(
        actual,
        torch.tensor(rates, dtype=actual.dtype).expand_as(actual),
        rtol=1e-6,
        atol=1e-6,
    )
    return model, rates


def draws(model, size, events=4):
    gaps, marks = [], []
    for start in range(0, size, 128):
        batch = history(min(128, size - start))
        with torch.no_grad():
            result = model.simulate(batch, num_events_to_simulate=events)
        assert result.valid_event_mask.all()
        gaps.append(result.time_delta_seqs.detach().numpy().astype(float))
        marks.append(result.type_seqs.numpy())
    return np.concatenate(gaps), np.concatenate(marks)


def distribution_check(values, distribution="expon", alpha=1e-4):
    values = np.asarray(values).ravel()
    statistic = float(kstest(values, distribution).statistic)
    bound = math.sqrt(math.log(2 / alpha) / (2 * len(values)))
    return dict(
        n=len(values),
        mean=float(values.mean()),
        ks_distance=statistic,
        dkw_bound=bound,
        alpha=alpha,
        passed=bool(statistic <= bound),
        zero_fraction=float((values == 0).mean()),
    )


def self_correcting_oracle(rng, count):
    """One total integrated hazard inversion in float64, then independent mark draw."""
    initial = history()
    t0 = float(initial.time_seqs[0, -1])
    marks = initial.type_seqs[0].numpy()
    penalty = ALPHA[:, marks].sum(axis=1)
    rates = np.exp(MU * t0 - penalty)
    times, types = [], []
    for _ in range(count):
        exponential = rng.exponential()

        def hazard(dt):
            return float(np.sum(rates * np.expm1(MU * dt) / MU))

        bound = 1.0
        while hazard(bound) < exponential:
            bound *= 2
        dt = brentq(lambda s: hazard(s) - exponential, 0, bound, xtol=1e-12)
        probability = rates * np.exp(MU * dt)
        probability /= probability.sum()
        times.append(dt)
        types.append(int(rng.choice(2, p=probability)))
    return np.array(times), np.array(types)


def validate_self_correcting(output, size):
    model = create_model("SelfCorrecting", output, mu=MU.tolist(), alpha=ALPHA.tolist())
    torch.manual_seed(20261016)
    gaps, marks = draws(model, size)
    initial = history()
    t = np.full(size, float(initial.time_seqs[0, -1]))
    penalty = np.tile(ALPHA[:, initial.type_seqs[0].numpy()].sum(axis=1), (size, 1))
    z_values, residuals, variances = [], [], []
    for step in range(gaps.shape[1]):
        rate = np.exp(t[:, None] * MU - penalty)
        z_values.append(np.sum(rate * np.expm1(gaps[:, step, None] * MU) / MU, axis=1))
        t += gaps[:, step]
        intensity = np.exp(t[:, None] * MU - penalty)
        p = intensity[:, 0] / intensity.sum(axis=1)
        residuals.append((marks[:, step] == 0) - p)
        variances.append(p * (1 - p))
        penalty += ALPHA[:, marks[:, step]].T
    check = distribution_check(np.concatenate(z_values))
    mark_z = float(np.sum(residuals) / np.sqrt(np.sum(variances)))
    oracle, _ = self_correcting_oracle(np.random.default_rng(20261017), size)
    from scipy.stats import ks_2samp

    distance = float(ks_2samp(gaps[:, 0], oracle).statistic)
    limit = math.sqrt(2 * math.log(4 / 1e-4) / size)
    passed = check["passed"] and abs(mark_z) < 6 and distance <= limit
    expected = np.exp(
        MU * (float(initial.time_seqs[0, -1]) + 0.4)
        - ALPHA[:, initial.type_seqs[0].numpy()].sum(axis=1)
    )
    actual = (
        model.compute_intensities_at_sample_dtimes(
            **initial.to_mapping(),
            sample_dtimes=torch.full((1, 3, 1), 0.4),
            compute_last_step_only=True,
        )
        .detach()
        .numpy()
        .ravel()
    )
    return dict(
        native_simulator=dict(
            passed=bool(passed),
            hazard_rescaling=check,
            conditional_mark_z=mark_z,
            independent_numpy_ks=distance,
            independent_numpy_bound=limit,
        ),
        post_event_intensity_api=dict(
            passed=bool(np.allclose(actual, expected, rtol=1e-6)),
            expected=expected.tolist(),
            actual=actual.tolist(),
            issue="API omits the correction from the conditioning event; native simulate includes it",
        ),
    )


def run(output, size=4096):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="in_progress",
        code=code_provenance(__file__),
        sequences=size,
        future_events=4,
        checks={},
        scope="Independent distribution references; known mismatches reported without changing scientific calculations",
    )
    report["checks"]["SelfCorrecting"] = validate_self_correcting(output, size)
    for name in POISSON_MODELS:
        model, rates = constant_poisson_model(name, output)
        torch.manual_seed(20261018)
        gaps, marks = draws(model, size)
        check = distribution_check(gaps * rates.sum())
        mark_z = float(
            ((marks == 0).sum() - marks.size / 2) / math.sqrt(marks.size / 4)
        )
        check.update(
            mark_z=mark_z,
            passed=check["passed"] and abs(mark_z) < 6,
            rates=rates.tolist(),
            initial_window=20.0,
            expected_probability_beyond_initial_window=math.exp(-20 * rates.sum()),
        )
        report["checks"][name] = check
        print(name, check, flush=True)
        write_json(output / "report.json", report)
    for case, initial_window, proposals, oversample in (
        ("small_initial_window", 0.25, 32, 5.0),
        ("many_rejections", 20.0, 1, 100.0),
    ):
        model, rates = constant_poisson_model("NHP", output, initial_window)
        sampler = model.get_event_sampler()
        sampler.num_exp = proposals
        sampler.over_sample_rate = oversample
        torch.manual_seed(20261019)
        gaps, _ = draws(model, size, events=1)
        check = distribution_check(gaps * rates.sum())
        check.update(
            initial_window=initial_window,
            initial_window_atom_fraction=float((gaps == initial_window).mean()),
            expected_poisson_tail=math.exp(-initial_window * rates.sum()),
            num_exp=proposals,
            oversample=oversample,
        )
        report["checks"][case] = check
    model = create_model("IntensityFree", output)
    zero_linear(model.linear)
    zero_linear(model.mark_linear)
    torch.manual_seed(20261020)
    gaps, marks = draws(model, size)
    check = distribution_check(
        (np.log(gaps) - model.mean_log_inter_time) / model.std_log_inter_time, "norm"
    )
    check.update(
        reference="lognormal independent of history, not Poisson",
        mark_zero_fraction=float((marks == 0).mean()),
    )
    report["checks"]["IntensityFree"] = check
    from new_ltpp.models.implementations.fullynn import CumulHazardFunctionNetwork

    torch.manual_seed(20261021)
    hazard = CumulHazardFunctionNetwork(hidden_size=4, num_event_types=2)
    _, base = hazard(torch.zeros(1, 1, 1, 4), torch.full((1, 1, 1), 0.5))
    _, repeated = hazard(torch.zeros(4, 1, 1, 4), torch.full((4, 1, 1), 0.5))
    _, grid = hazard(torch.zeros(1, 1, 7, 4), torch.full((1, 1, 7), 0.5))
    report["checks"]["FullyNN"] = dict(
        batch_invariance_passed=bool(torch.allclose(base[0], repeated[0])),
        sample_grid_invariance_passed=bool(
            torch.allclose(base[0, 0, 0], grid[0, 0, 0])
        ),
        batch_intensity_ratio=(repeated[0] / base[0]).detach().tolist(),
        sample_grid_ratio=(grid[0, 0, 0] / base[0, 0, 0]).detach().tolist(),
        poisson_case="Not asserted: tanh cumulative hazard is bounded for finite weights, incompatible with positive constant hazard on an unbounded horizon",
        cause="pointwise sum differentiation; no batch/time/grid scaling",
    )
    hazard_at_zero, _ = hazard(torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1))
    linear = hazard.layer_dense_3[0]
    upper = torch.nn.functional.softplus(linear.weight.sum(dim=1) + linear.bias)
    report["checks"]["FullyNN"].update(
        cumulative_hazard_at_zero=hazard_at_zero.detach().flatten().tolist(),
        finite_cumulative_hazard_upper_bound=upper.detach().tolist(),
        lower_bound_survival_at_infinity_if_centered=float(
            torch.exp(-(upper.sum() - hazard_at_zero.sum()))
        ),
    )
    # Deterministic small-growth inversion regression, independent of a lucky RNG draw.
    small = create_model(
        "SelfCorrecting", output, mu=[1e-5, 1e-5], alpha=[[0.0, 0.0], [0.0, 0.0]]
    )
    from unittest.mock import patch

    def fixed_exponential(tensor, *args, **kwargs):
        return tensor.fill_(0.001)

    with patch.object(torch.Tensor, "exponential_", fixed_exponential):
        simulated = small.simulate(history(), num_events_to_simulate=1)
    expected = float(
        np.log1p(0.001 * 1e-5 / np.exp(1e-5 * float(history().time_seqs[0, -1]))) / 1e-5
    )
    report["checks"]["SelfCorrecting_small_growth"] = dict(
        expected_delay=expected,
        actual_delay=float(simulated.time_delta_seqs[0, 0]),
        passed=bool(
            np.isclose(
                float(simulated.time_delta_seqs[0, 0]), expected, rtol=1e-5, atol=1e-8
            )
        ),
        cause="log(1+x) float32 cancellation; log1p would retain the increment",
    )
    report["status"] = "completed_with_findings"
    write_json(output / "report.json", report)
    print(json.dumps(report["checks"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/independent-simulators")
    )
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
