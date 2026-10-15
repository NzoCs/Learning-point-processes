"""Prespecified CPU calibration through the current conditional accumulator.

Keep the production sampler configuration, observation quotas, termination
flags, pair IDs, global normalization and blocked MMD. Oracle scope is explicit:
NumPy/SciPy for parametric/analytic models; internal H0 plus a separate Poisson
oracle control for the remaining neural models. Never silently discard errors.
"""

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import softmax
from scipy.stats import binomtest

from new_ltpp.configs import DataConfig, ModelConfig, StatisticalTestConfig
from new_ltpp.data.preprocess import TPPDataModule
from new_ltpp.evaluation.accumulators.statistical_metrics_accumulator import (
    StatisticalTestAccumulator,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
    split_history_future,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.conditional_mmd_test import (
    paired_joint_statistic,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.global_conditional_mmd_test import (
    copy_rows,
)
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.shared_types import Batch
from scripts.conditional_calibration import conditional_hawkes_reference
from scripts.conditional_integration_suite import make_splits
from scripts.extended_scientific_validation import code_provenance, fingerprint
from scripts.independent_simulator_validation import zero_linear
from scripts.integration_suite import CATALOGUE, FIXTURES, write_json
from scripts.scientific_calibration import wilson_interval
from scripts.stopped_conditional_calibration import encoded_last_state, numpy_hazard

ARCHIVE = Path("docs/validation/global-pipeline-20261014")
STOPPED = ("FullyNN", "RMTPP")
POISSON_CONTROLS = ("ANHN", "ANHP", "NHP", "ODETPP", "SAHP", "THP")
SCENARIOS = {"quota3": (3, (16, 11, 5)), "quota6": (6, (64, 63, 1))}
FAMILY_SIZE = 36  # 12 main model/scenario cells + 6 Poisson-control/scenario cells, times 2 scenarios.


def model_and_config(name, output):
    import new_ltpp.models  # noqa: F401 -- model registry

    raw = json.loads((FIXTURES / "config.json").read_text())
    raw["model_config"]["model_id"] = name
    splits = make_splits()
    for split, key in (
        ("train", "train_dir"),
        ("dev", "valid_dir"),
        ("test", "test_dir"),
    ):
        path = output / "fit-source" / f"{split}.json"
        write_json(path, splits[split])
        raw["data_config"][key] = str(path)
    info = TPPDataModule(DataConfig.model_validate(raw["data_config"])).get_data_info()
    torch.manual_seed(raw["training_config"]["seed"])
    model = ModelFactory.create_model_by_name(
        name, ModelConfig.model_validate(raw["model_config"]), info, output / "model"
    )
    source = ARCHIVE / f"{name}_conditional_checkpoint_after.json"
    state = json.loads(source.read_text())["state_dict"]
    model.load_state_dict(
        {
            k: torch.tensor(
                v["values"], dtype=getattr(torch, v["dtype"].split(".")[-1])
            ).reshape(v["shape"])
            for k, v in state.items()
        },
        strict=True,
    )
    model.eval()
    model.requires_grad_(False)
    model._simulator = Simulator(model)
    config = StatisticalTestConfig.model_validate(
        dict(raw["statistical_test_config"], n_samples=99)
    )
    return (
        model,
        config,
        dict(
            source=str(source),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            model_state_sha256=state_hash(model),
            independently_fitted=True,
            training_seeds=[1701, 1702],
            effective_model_config=raw["model_config"],
            data_info=dict(info),
        ),
    )


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def histories_for_trial(n, rng, trial):
    # Two dense/sparse contexts with varying gaps and marks, fixed two-event prefix.
    gaps = rng.uniform(0.05, 0.3, (n, 2))
    gaps[1::2] *= 5
    times = np.cumsum(gaps, axis=1).astype(np.float32)
    marks = rng.integers(0, 2, (n, 2))
    marks[::4] = 0
    marks[1::4] = 1
    t = torch.tensor(times)
    return Batch(
        t,
        torch.diff(t, prepend=torch.zeros(n, 1)),
        torch.tensor(marks),
        torch.ones(n, 2, dtype=torch.bool),
        sequence_ids=[f"evaluation-{trial:06}-{i:06}" for i in range(n)],
    )


def analytic_continuations(model, histories, rng, quota, *, never_stop=False):
    """Independent draws/decoder/inverse, shared frozen Torch encoder where needed.

    Match the production float32 event clock and store original sampled gaps;
    do not replace them by differences of rounded absolute event times.
    """
    name = type(model).__name__
    n, length = histories.time_seqs.shape
    times = np.zeros((n, length + quota), dtype=np.float32)
    gaps = np.zeros_like(times)
    marks = np.full(times.shape, model.pad_token_id, dtype=np.int64)
    times[:, :length], gaps[:, :length], marks[:, :length] = (
        histories.time_seqs.numpy(),
        histories.time_delta_seqs.numpy(),
        histories.type_seqs.numpy(),
    )
    valid = np.zeros((n, quota), dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    for step in range(quota):
        rows = np.flatnonzero(~stopped)
        if not len(rows):
            break
        prefix = Batch(
            torch.tensor(times[rows, : length + step]),
            torch.tensor(gaps[rows, : length + step]),
            torch.tensor(marks[rows, : length + step]),
            torch.ones(len(rows), length + step, dtype=torch.bool),
        )
        hidden = encoded_last_state(model, prefix) if name in STOPPED else None
        context = (
            model.forward(prefix.time_delta_seqs, prefix.type_seqs)[:, -1]
            .detach()
            .double()
            .numpy()
            if name == "IntensityFree"
            else None
        )
        for k, row in enumerate(rows):
            if name in STOPPED:
                hazard, marked, limit = numpy_hazard(model, hidden[k])
                target = rng.exponential()
                if never_stop and np.isfinite(limit):
                    if limit <= 0:
                        raise RuntimeError("Next event has zero probability")
                    target = -np.logaddexp(-target, np.log(-np.expm1(-target)) - limit)
                if target >= limit:
                    stopped[row] = True
                    continue
                high = 1.0
                while hazard(high) < target:
                    high *= 2
                    if not np.isfinite(high) or high > 1e100:
                        raise RuntimeError(
                            "Independent inverse cannot bracket the event"
                        )
                dt = brentq(
                    lambda t: hazard(t) - target, 0.0, high, xtol=1e-12, rtol=1e-12
                )
                intensity = marked(dt)
                mark = rng.choice(2, p=intensity / intensity.sum())
            elif name == "IntensityFree":
                h = context[k]

                def linear(layer):
                    return (
                        layer.weight.detach().double().numpy() @ h
                        + layer.bias.detach().double().numpy()
                    )

                params = linear(model.linear)
                count = model.num_mix_components
                component = rng.choice(count, p=softmax(params[2 * count :]))
                log_t = rng.normal(
                    params[component], np.exp(np.clip(params[count + component], -5, 3))
                )
                dt = np.exp(
                    model.mean_log_inter_time + model.std_log_inter_time * log_t
                )
                mark = rng.choice(2, p=softmax(linear(model.mark_linear)))
            else:
                raise ValueError(name)
            if not np.isfinite(dt) or dt <= 0:
                raise RuntimeError(
                    "Independent oracle returned an invalid positive delay"
                )
            times[row, length + step] = times[row, length + step - 1] + np.float32(dt)
            gaps[row, length + step], marks[row, length + step] = dt, mark
            valid[row, step] = True
    relative = times[:, length:] - times[:, length - 1 : length]
    relative[~valid] = 0
    return Batch(
        torch.tensor(relative),
        torch.tensor(gaps[:, length:]),
        torch.tensor(marks[:, length:]),
        torch.tensor(valid),
        no_next_event_mask=torch.tensor(stopped),
    )


def parametric_continuations(model, h, rng, quota):
    name = type(model).__name__
    if name == "Hawkes":
        return conditional_hawkes_reference(
            rng,
            h,
            model.mu_pos.detach().numpy(),
            model.alpha_pos.detach().numpy(),
            model.beta_pos.detach().numpy(),
            events=quota,
        )
    if name == "Poisson":
        return poisson_continuations(h, rng, quota, model.rates.detach().numpy())
    mu = model.mu.detach().clamp(min=model.eps).double().numpy()
    alpha = model.alpha.detach().double().numpy()
    rows_t, rows_m = [], []
    for row in range(len(h.time_seqs)):
        current = float(h.time_seqs[row, -1])
        penalty = alpha[:, h.type_seqs[row].numpy()].sum(1)
        t, m, elapsed = [], [], 0.0
        for _ in range(quota):
            rate = np.exp(mu * current - penalty)
            target = rng.exponential()

            def integrated(dt):
                return np.sum(rate * np.expm1(mu * dt) / mu)

            high = 1.0
            while integrated(high) < target:
                high *= 2
                if high > 1e12:
                    raise RuntimeError("SelfCorrecting oracle cannot bracket inverse")
            dt = brentq(lambda x: integrated(x) - target, 0.0, high, xtol=1e-12)
            intensity = rate * np.exp(mu * dt)
            mark = int(rng.choice(2, p=intensity / intensity.sum()))
            elapsed += dt
            current += dt
            penalty += alpha[:, mark]
            t.append(elapsed)
            m.append(mark)
        rows_t.append(t)
        rows_m.append(m)
    return from_arrays(rows_t, rows_m)


def from_arrays(times, marks):
    t = torch.tensor(np.asarray(times), dtype=torch.float32)
    return Batch(
        t,
        torch.diff(t, prepend=torch.zeros(len(t), 1)),
        torch.tensor(np.asarray(marks), dtype=torch.long),
        torch.ones_like(t, dtype=torch.bool),
    )


def poisson_continuations(h, rng, quota, rates):
    shape = (len(h.time_seqs), quota)
    return from_arrays(
        np.cumsum(rng.exponential(1 / np.sum(rates), shape), axis=1),
        rng.choice(2, shape, p=rates / np.sum(rates)),
    )


def poisson_control(model):
    """Separate boundary-case control; never overwrite the fitted model."""
    model = copy.deepcopy(model)
    name = type(model).__name__
    with torch.no_grad():
        for layer in model.modules():
            if hasattr(layer, "log_beta"):
                layer.log_beta.zero_()
        if name in ("THP", "SAHP"):
            zero_linear(model.layer_intensity_hidden)
            model.softplus.log_beta.zero_()
            if name == "SAHP":
                zero_linear(model.mu)
                zero_linear(model.eta)
            else:
                model.factor_intensity_base.zero_()
                model.factor_intensity_decay.zero_()
        else:
            zero_linear(model.layer_intensity)
            if name == "ANHP":
                model.softplus.log_beta.zero_()
    model._simulator = Simulator(model)
    return model


def observed_future(model, h, rng, quota, seed, control=False, never_stop=False):
    name = type(model).__name__
    if control:
        return poisson_continuations(h, rng, quota, np.full(2, math.log(2)))
    if name in STOPPED or name == "IntensityFree":
        return analytic_continuations(model, h, rng, quota, never_stop=never_stop)
    if name in ("Poisson", "Hawkes", "SelfCorrecting"):
        return parametric_continuations(model, h, rng, quota)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        return model.simulate(h, num_events_to_simulate=quota)


def holdout(h, future, quota, stopped):
    """Use the production split function on complete identified source records."""
    width = max(quota, future.time_seqs.shape[1])
    times = torch.zeros(len(h.time_seqs), width)
    marks = torch.full_like(times, 2, dtype=torch.long)
    mask = torch.zeros_like(times, dtype=torch.bool)
    k = future.time_seqs.shape[1]
    times[:, :k] = torch.where(
        future.valid_event_mask, future.time_seqs + h.time_seqs[:, -1:], 0
    )
    marks[:, :k], mask[:, :k] = future.type_seqs, future.valid_event_mask
    times, marks, mask = (
        torch.cat((h.time_seqs, times), 1),
        torch.cat((h.type_seqs, marks), 1),
        torch.cat((h.valid_event_mask, mask), 1),
    )
    raw = Batch(
        times,
        torch.diff(times, prepend=torch.zeros(len(times), 1)).masked_fill(~mask, 0),
        marks,
        mask,
        no_next_event_mask=future.no_next_event_mask if stopped else None,
        sequence_ids=h.sequence_ids,
    )
    return split_history_future(
        raw, h.time_seqs.shape[1], quota, allow_stopped=stopped, pad_token_id=2
    )


def evaluate(model, config, histories, observed, sizes, quota, seed, dense_check=False):
    stopped = type(model).__name__ in STOPPED
    accumulator = StatisticalTestAccumulator(
        config,
        conditional=True,
        complete_stopped=stopped,
        stopped_event_budget=quota if stopped else None,
    )
    torch.manual_seed(seed)
    offset = 0
    generated, prepared = [], []
    for size in sizes:
        indices = list(range(offset, offset + size))
        h, x = holdout(
            copy_rows(histories, indices), copy_rows(observed, indices), quota, stopped
        )
        with torch.no_grad():
            y = model.simulate(h, num_events_to_simulate=quota)
        y.sequence_ids = h.sequence_ids
        before = torch.get_rng_state().clone()
        accumulator.update(x, y, histories=h, pair_ids=h.sequence_ids)
        if not torch.equal(before, torch.get_rng_state()):
            raise AssertionError("Statistical test consumed trajectory RNG")
        generated.append(y)
        prepared.append((h, x, y))
        offset += size
    result = accumulator.compute()
    if dense_check:
        h, x, y = accumulator.test._pooled_observations()
        dense = ConditionalMMDTwoSampleTest(
            accumulator.test.kernel,
            config.n_samples,
            stopped_event_budget=quota if stopped else None,
        ).joint_kernel_matrix(h, x, y)
        n = len(h.time_seqs)
        rows = torch.arange(n)
        rng = torch.Generator().manual_seed(result["global_null_seed"])
        swaps = torch.randint(2, (config.n_samples, n), generator=rng)
        values = torch.stack(
            [paired_joint_statistic(dense, rows, rows + n)]
            + [
                paired_joint_statistic(dense, rows + n * s, rows + n * (1 - s))
                for s in swaps
            ]
        )
        torch.testing.assert_close(
            torch.tensor(
                [
                    result["global_observed_statistic"],
                    *result["global_permuted_statistic"],
                ],
                dtype=torch.float64,
            ),
            values,
            rtol=1e-10,
            atol=1e-12,
        )
    return result, generated, prepared


def run(name, scenario, output, *, h0_trials=None, power_trials=200, seed=2026101523):
    torch.set_num_threads(1)
    output = Path(output)
    if (output / "plan.json").exists():
        raise FileExistsError("Use a fresh output folder")
    quota, sizes = SCENARIOS[scenario]
    h0_trials = (1000 if name in STOPPED else 500) if h0_trials is None else h0_trials
    if min(h0_trials, power_trials) < 1:
        raise ValueError("Positive trial counts required")
    # Plan is written before model/evaluation draws; no result-based tuning.
    cases = ["H0", "twice_speed", "context_marks"] + (
        ["never_stop", "always_stop"] if name in STOPPED else []
    )
    if name in POISSON_CONTROLS:
        cases.append("poisson_H0")
    report = dict(
        status="in_progress",
        model=name,
        scenario=scenario,
        seed=seed,
        code=code_provenance(__file__),
        event_quota=quota,
        batch_sizes=list(sizes),
        history_events=2,
        marks=2,
        H0_trials=h0_trials,
        H1_trials=power_trials,
        cases=cases,
        alpha=0.05,
        h0_excess_family_threshold=0.01 / FAMILY_SIZE,
        h0_family_size=FAMILY_SIZE,
        protocol="exact_production_holdout_accumulator_global_signature_v1",
        observation=(
            "complete_stopped_event_budget_v1"
            if name in STOPPED
            else "pairwise_truncation_recomputed_v1"
        ),
        reference_scope=(
            "Independent NumPy/SciPy decoder/RNG with shared Torch encoder"
            if name in (*STOPPED, "IntensityFree")
            else (
                "Independent NumPy parametric oracle"
                if name in ("Poisson", "Hawkes", "SelfCorrecting")
                else "Internal frozen-model H0; separate independent Poisson boundary control; does not certify fitted nonconstant neural sampler"
            )
        ),
        results=[],
        errors=[],
        scope="CPU only; independently fitted archived checkpoint; two quotas; no fitting uncertainty or universal calibration guarantee",
    )
    write_json(output / "plan.json", report)
    model, config, fit = model_and_config(name, output)
    report["fit"], report["kernel_config"] = fit, config.model_dump()
    original_hash = state_hash(model)
    control = poisson_control(model) if name in POISSON_CONTROLS else None
    if control is not None:
        report["poisson_control_state_sha256"] = state_hash(control)
        report["poisson_control_mark_rates"] = [math.log(2)] * 2
    started = time.monotonic()
    for case_id, case in enumerate(cases):
        count = h0_trials if case.endswith("H0") else power_trials
        values, draws, counts = (
            [],
            [],
            {
                key: 0
                for key in (
                    "observed_rows",
                    "observed_empty",
                    "observed_stopped",
                    "simulated_rows",
                    "simulated_empty",
                    "simulated_stopped",
                )
            },
        )
        digest = hashlib.sha256()
        active = control if case == "poisson_H0" else model
        for trial in range(count):
            trial_seed = (
                seed
                + CATALOGUE.index(name) * 100000000
                + list(SCENARIOS).index(scenario) * 10000000
                + case_id * 1000000
                + trial * 1000
            )
            try:
                h = histories_for_trial(
                    sum(sizes), np.random.default_rng(trial_seed), trial
                )
                if case == "always_stop":
                    t = torch.zeros(len(h.time_seqs), quota)
                    x = Batch(
                        t,
                        t.clone(),
                        torch.full_like(t, 2, dtype=torch.long),
                        torch.zeros_like(t, dtype=torch.bool),
                        no_next_event_mask=torch.ones(len(t), dtype=torch.bool),
                    )
                else:
                    x = observed_future(
                        active,
                        h,
                        np.random.default_rng(trial_seed + 101),
                        quota,
                        trial_seed + 201,
                        control=case == "poisson_H0",
                        never_stop=case == "never_stop",
                    )
                if case == "twice_speed":
                    x.time_seqs *= 0.5
                    x.time_delta_seqs *= 0.5
                elif case == "context_marks":
                    x.type_seqs = torch.where(
                        x.valid_event_mask,
                        h.type_seqs[:, -1:].expand_as(x.type_seqs),
                        x.type_seqs,
                    )
                result, generated, prepared = evaluate(
                    active,
                    config,
                    h,
                    x,
                    sizes,
                    quota,
                    trial_seed + 301,
                    dense_check=trial == 0,
                )
                if trial == 0:
                    replay, _, _ = evaluate(
                        active, config, h, x, sizes, quota, trial_seed + 301
                    )
                    if result != replay:
                        raise AssertionError("Exact repetition failed")
                    write_json(output / f"{case}_first_trial.json", result)
                values.append(result["pooled_p_value"])
                draws.append(
                    dict(
                        seed=trial_seed,
                        observed=result["global_observed_statistic"],
                        null=result["global_permuted_statistic"],
                        null_seed=result["global_null_seed"],
                    )
                )
                for label, batches in (
                    ("observed", [b[1] for b in prepared]),
                    ("simulated", generated),
                ):
                    for batch in batches:
                        counts[label + "_rows"] += len(batch.time_seqs)
                        counts[label + "_empty"] += int(
                            (~batch.valid_event_mask.any(-1)).sum()
                        )
                        counts[label + "_stopped"] += (
                            0
                            if batch.no_next_event_mask is None
                            else int(batch.no_next_event_mask.sum())
                        )
                        digest.update(fingerprint(batch).encode())
                        if batch.no_next_event_mask is not None:
                            digest.update(batch.no_next_event_mask.numpy().tobytes())
                if (trial + 1) % 50 == 0:
                    print(name, scenario, case, trial + 1, "/", count, flush=True)
            except Exception as error:
                report["errors"].append(
                    dict(case=case, trial=trial, seed=trial_seed, error=repr(error))
                )
                write_json(output / "report.json", report)
                # Keep failure visible and continue the predeclared experiment;
                # never report a filtered successful subset as calibrated.
                print(name, scenario, case, trial, "ERROR", repr(error), flush=True)
        rejected = sum(p <= 0.05 for p in values)
        is_h0 = case.endswith("H0")
        excess = (
            float(binomtest(rejected, len(values), 0.05, alternative="greater").pvalue)
            if is_h0 and values
            else None
        )
        row = dict(
            hypothesis=case,
            planned_trials=count,
            completed_trials=len(values),
            failed_trials=count - len(values),
            rejections=rejected,
            rejection_rate=rejected / len(values) if values else None,
            wilson_95=wilson_interval(rejected, len(values)) if values else None,
            excess_binomial_p=excess,
            h0_excess_detected=excess is not None and excess < 0.01 / FAMILY_SIZE,
            calibration_assessable=is_h0 and len(values) == count,
            p_values=values,
            counts=counts,
            data_sha256=digest.hexdigest(),
            reference_scope=(
                report["reference_scope"]
                if case != "poisson_H0"
                else "Independent NumPy marked Poisson with same production sampler configuration"
            ),
        )
        write_json(output / f"{case}_draws.json", draws)
        report["results"].append(row)
        write_json(output / "report.json", report)
    if state_hash(model) != original_hash:
        raise AssertionError("Fitted model parameters changed")
    report["model_frozen_verified"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "completed" if not report["errors"] else "completed_with_errors"
    report["no_h0_excess_detected"] = not any(
        r["h0_excess_detected"] for r in report["results"]
    )
    write_json(output / "report.json", report)
    if report["errors"] or not report["no_h0_excess_detected"]:
        raise AssertionError(
            "Prespecified campaign failed; all available results retained"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="name", choices=CATALOGUE, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h0-trials", type=int)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026101523)
    run(**vars(parser.parse_args()))
