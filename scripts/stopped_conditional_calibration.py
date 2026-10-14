"""CPU calibration of stopped learned continuations against a NumPy/SciPy oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binomtest

from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import (
    cached_paired_statistics,
    heterogeneous_histories,
    swap_orders,
)
from scripts.extended_scientific_validation import code_provenance, fingerprint
from scripts.independent_simulator_validation import create_model
from scripts.integration_suite import write_json
from scripts.scientific_calibration import wilson_interval

CHECKPOINTS = Path("docs/validation/unbounded-sampling-20261014")


def load_frozen_model(name, output, checkpoint=None):
    source = Path(checkpoint) if checkpoint else CHECKPOINTS / f"{name}_snapshot.json"
    model = create_model(name, output)
    if checkpoint:
        state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
    else:
        saved = json.loads(source.read_text())
        state = {
            k: torch.tensor(v["values"]).reshape(v["shape"])
            for k, v in saved["trained_state"].items()
        }
    model.load_state_dict(
        state,
        strict=True,
    )
    model.eval()
    model.requires_grad_(False)
    state_digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        state_digest.update(key.encode())
        state_digest.update(str((value.shape, value.dtype)).encode())
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return model, dict(
        source=str(source),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        independently_fitted_before_this_experiment=True,
        model_state_sha256=state_digest.hexdigest(),
    )


def numpy_hazard(model, hidden):
    """Independent elapsed-time decoder and derivatives, with a shared fixed encoder."""

    if not hasattr(model, "_calibration_numpy_weights"):
        model._calibration_numpy_weights = {
            id(layer): (
                layer.weight.detach().double().numpy(),
                layer.bias.detach().double().numpy(),
            )
            for layer in model.modules()
            if isinstance(layer, torch.nn.Linear)
        }
    weights = model._calibration_numpy_weights

    def linear(layer, x):
        weight, bias = weights[id(layer)]
        return weight @ x + bias

    if model.__class__.__name__ == "RMTPP":
        a = (
            linear(model.hidden_to_intensity_logits, hidden)
            + model.b_t.detach().double().numpy()[0]
        )
        w = model.w_t.detach().double().numpy()[0]
        cap = np.log(1e5)

        def marked(t):
            return np.exp(np.minimum(a + w * t, cap))

        def integrated(t):
            total = 0.0
            for start, slope in zip(a, w):
                if slope == 0:
                    total += np.exp(min(start, cap)) * t
                elif slope > 0:
                    split = max(0.0, (cap - start) / slope)
                    duration = min(t, split)
                    total += (
                        np.exp(min(start, cap)) * np.expm1(slope * duration) / slope
                    )
                    total += 1e5 * max(0.0, t - split)
                else:
                    plateau = max(0.0, (start - cap) / (-slope))
                    total += 1e5 * min(t, plateau)
                    duration = max(0.0, t - plateau)
                    total += (
                        np.exp(min(start, cap)) * np.expm1(slope * duration) / slope
                    )
            return total

        limit = (
            np.inf
            if np.any(w >= 0)
            else sum(
                np.exp(min(start, cap)) / (-slope)
                + 1e5 * max(0.0, (start - cap) / (-slope))
                for start, slope in zip(a, w)
            )
        )
        return integrated, marked, limit

    net = model.layer_intensity
    for p in net.parameters():
        if np.any(p.detach().numpy() < net.params_eps):
            raise ValueError(
                "Frozen FullyNN parameters must already satisfy its positivity constraint"
            )
    time_weight = weights[id(net.layer_dense_1)][0][:, 0]
    dense_weight = weights[id(net.layer_dense_2)][0]

    def decode(t):
        elapsed = linear(net.layer_dense_1, np.array([t]))
        value = np.tanh(linear(net.layer_dense_2, np.concatenate((hidden, elapsed))))
        derivative = (1 - value * value) * (
            dense_weight[:, len(hidden) :] @ time_weight
        )
        for layer in net.module_list:
            value = np.tanh(linear(layer, value))
            derivative = (1 - value * value) * (weights[id(layer)][0] @ derivative)
        z = linear(net.layer_dense_3[0], value)
        intensity = expit(z) * (weights[id(net.layer_dense_3[0])][0] @ derivative)
        return np.logaddexp(0.0, z).sum(), intensity

    zero = decode(0.0)[0]
    saturated = np.ones(len(hidden))
    for layer in net.module_list:
        saturated = np.tanh(linear(layer, saturated))
    limit = np.logaddexp(0.0, linear(net.layer_dense_3[0], saturated)).sum() - zero
    return lambda t: max(0.0, decode(t)[0] - zero), lambda t: decode(t)[1], limit


def encoded_last_state(model, batch):
    with torch.no_grad():
        if model.__class__.__name__ == "FullyNN":
            h = model.forward(batch.time_delta_seqs, batch.type_seqs)
        else:
            features = model.layer_type_emb(batch.type_seqs) + model.layer_temporal_emb(
                batch.time_seqs[..., None]
            )
            h, _ = model.layer_rnn(features)
    return h[:, -1].double().numpy()


def oracle_continuations(model, histories, rng, budget, *, never_stop=False):
    """NumPy RNG, analytic marked derivatives, SciPy roots; no production sampler calls."""
    n, initial = histories.time_seqs.shape
    times = np.zeros((n, initial + budget), dtype=np.float64)
    marks = np.full(times.shape, model.pad_token_id, dtype=np.int64)
    times[:, :initial] = histories.time_seqs.numpy()
    marks[:, :initial] = histories.type_seqs.numpy()
    mask = np.zeros((n, budget), dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    for step in range(budget):
        alive = np.flatnonzero(~stopped)
        if not len(alive):
            break
        prefix_t = torch.tensor(times[alive, : initial + step], dtype=torch.float32)
        prefix = Batch(
            prefix_t,
            torch.diff(prefix_t, prepend=torch.zeros(len(alive), 1)),
            torch.tensor(marks[alive, : initial + step]),
            torch.ones_like(prefix_t, dtype=torch.bool),
        )
        hidden = encoded_last_state(model, prefix)
        for index, row in enumerate(alive):
            hazard, marked, limit = numpy_hazard(model, hidden[index])
            target = rng.exponential()
            if never_stop and np.isfinite(limit):
                if limit <= 0:
                    raise RuntimeError("Cannot condition on an impossible next event")
                target = -np.logaddexp(-target, np.log(-np.expm1(-target)) - limit)
            if target >= limit:
                stopped[row] = True
                continue
            high = 1.0
            while hazard(high) < target:
                high *= 2
                if not np.isfinite(high) or high > 1e100:
                    raise RuntimeError("Oracle root cannot be bracketed")
            delay = brentq(
                lambda t: hazard(t) - target, 0.0, high, xtol=1e-12, rtol=1e-12
            )
            intensity = marked(delay)
            if (
                not np.isfinite(intensity).all()
                or np.any(intensity < 0)
                or intensity.sum() <= 0
            ):
                raise RuntimeError("Invalid oracle marked intensity")
            times[row, initial + step] = times[row, initial + step - 1] + delay
            marks[row, initial + step] = rng.choice(
                model.num_event_types, p=intensity / intensity.sum()
            )
            mask[row, step] = True
    relative = times[:, initial:] - times[:, initial - 1 : initial]
    relative[~mask] = 0
    deltas = np.diff(relative, prepend=np.zeros((n, 1)), axis=1)
    deltas[~mask] = 0
    return Batch(
        torch.tensor(relative, dtype=torch.float32),
        torch.tensor(deltas, dtype=torch.float32),
        torch.tensor(marks[:, initial:]),
        torch.tensor(mask),
        no_next_event_mask=torch.tensor(stopped),
    )


def empty_continuations(n, budget):
    t = torch.zeros(n, budget)
    return Batch(
        t,
        t.clone(),
        torch.full((n, budget), 2, dtype=torch.long),
        torch.zeros_like(t, dtype=torch.bool),
        no_next_event_mask=torch.ones(n, dtype=torch.bool),
    )


def batch_fingerprint(batch):
    return (
        fingerprint(batch)
        + hashlib.sha256(batch.no_next_event_mask.numpy().tobytes()).hexdigest()
    )


def run(
    name,
    output,
    trials=500,
    power_trials=200,
    permutations=99,
    seed=20261022,
    checkpoint=None,
    h0_family_size=8,
):
    if min(trials, power_trials, permutations, h0_family_size) < 1:
        raise ValueError("Trial and permutation counts must be positive")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a fresh calibration output directory")
    model, fit = load_frozen_model(name, output, checkpoint)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    budget, sizes = 6, [16, 11, 5]
    tests = {
        protocol: ConditionalMMDTwoSampleTest(
            SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2),
            permutations,
            stopped_event_budget=budget if protocol == "complete_stopped" else None,
        )
        for protocol in ("historical_truncation", "complete_stopped")
    }
    report = dict(
        status="in_progress",
        model=name,
        code=code_provenance(__file__),
        fit=fit,
        seed=seed,
        trials_H0=trials,
        power_trials_per_alternative=power_trials,
        permutations=permutations,
        batch_sizes=sizes,
        event_budget=budget,
        marks=2,
        alpha=0.05,
        results=[],
        scope="Frozen separately fitted model; NumPy/SciPy elapsed-time decoder and simulator with shared frozen Torch history encoder. Independent histories/pairs/batches; internal training uncertainty and CUDA excluded.",
        acceptance=f"H0 one-sided binomial excess test at Bonferroni 0.01/{h0_family_size}; power reported separately, no post-hoc tuning",
    )
    write_json(output / "report.json", report)
    for case_id, case in enumerate(("H0", "never_stop", "always_stop")):
        count = trials if case == "H0" else power_trials
        values = {protocol: dict(single=[], aggregated=[]) for protocol in tests}
        hashes = hashlib.sha256()
        totals = dict(
            observed_rows=0,
            observed_empty=0,
            observed_stopped=0,
            simulated_rows=0,
            simulated_empty=0,
            simulated_stopped=0,
        )
        for trial in range(count):
            for test in tests.values():
                test.reset_accumulators()
            for block, n in enumerate(sizes):
                histories = heterogeneous_histories(n)
                sample_seed = seed + case_id * 10000000 + trial * 1000 + block * 10
                torch.manual_seed(sample_seed + 100000000)
                with torch.no_grad():
                    simulated = model.simulate(histories, num_events_to_simulate=budget)
                observed = (
                    empty_continuations(n, budget)
                    if case == "always_stop"
                    else oracle_continuations(
                        model,
                        histories,
                        np.random.default_rng(sample_seed + 200000000),
                        budget,
                        never_stop=case == "never_stop",
                    )
                )
                orders = swap_orders(n, permutations, sample_seed + 300000000)
                for label, batch in (("observed", observed), ("simulated", simulated)):
                    totals[label + "_rows"] += n
                    totals[label + "_empty"] += int(
                        (~batch.valid_event_mask.any(-1)).sum()
                    )
                    totals[label + "_stopped"] += int(batch.no_next_event_mask.sum())
                hashes.update(
                    (
                        fingerprint(histories)
                        + batch_fingerprint(observed)
                        + batch_fingerprint(simulated)
                    ).encode()
                )
                for protocol, test in tests.items():
                    stats = cached_paired_statistics(
                        test, histories, observed, simulated, orders
                    )
                    if trial == 0:
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(sample_seed + 300000000)
                            actual = test.compute_statistics(
                                observed,
                                simulated,
                                histories=histories,
                                pair_ids=[f"{case}/{block}/{i}" for i in range(n)],
                                accumulate=False,
                            )
                        torch.testing.assert_close(
                            stats[0], actual["observed_statistic"], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            stats[1:], actual["permuted_statistics"], rtol=0, atol=0
                        )
                    test._accumulate(stats[0], stats[1:])
                    if block == 0:
                        values[protocol]["single"].append(
                            float(monte_carlo_p_value(stats[0], stats[1:]))
                        )
            for protocol, test in tests.items():
                values[protocol]["aggregated"].append(float(test.get_final_p_value()))
            if (trial + 1) % 50 == 0:
                print(f"{name}/{case}: {trial + 1}/{count}", flush=True)
        for protocol, methods in values.items():
            for method, pvalues in methods.items():
                rejected = sum(p <= 0.05 for p in pvalues)
                excess = (
                    binomtest(rejected, count, 0.05, alternative="greater").pvalue
                    if case == "H0"
                    else None
                )
                report["results"].append(
                    dict(
                        hypothesis=case,
                        protocol=protocol,
                        method=method,
                        trials=count,
                        rejections=rejected,
                        rejection_rate=rejected / count,
                        wilson_95=wilson_interval(rejected, count),
                        excess_binomial_p=excess,
                        h0_excess_detected=(
                            bool(excess < 0.01 / h0_family_size)
                            if excess is not None
                            else None
                        ),
                        p_values=pvalues,
                        data_sha256=hashes.hexdigest(),
                        termination_counts=totals,
                        public_api_all_draws_verified_on_first_trial=True,
                    )
                )
                print(name, case, protocol, method, rejected, count, flush=True)
        write_json(output / "report.json", report)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
    report.update(status="completed", model_frozen_verified=True)
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("FullyNN", "RMTPP"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint fitted on independent training data"
    )
    args = vars(parser.parse_args())
    args["name"] = args.pop("model")
    run(**args)
