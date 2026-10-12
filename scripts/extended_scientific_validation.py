"""Opt-in marked and learned-model calibration, with disjoint fitting data.

Internal learned-model H0 checks exchangeability conditional on one frozen fit.
Held-out target/model rejection measures fit, not a type-I error rate.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from new_ltpp.configs import ModelConfig
from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel
from new_ltpp.evaluation.statistical_testing.point_process_kernels.space_kernels import (
    RBFKernel,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    MMDTwoSampleTest,
)
from new_ltpp.models.model_factory import ModelFactory
from new_ltpp.models.simulation.simulator import Simulator
from new_ltpp.shared_types import Batch
from scripts.compare_truncation import paired_statistics, permutation_p_value
from scripts.scientific_calibration import wilson_interval

MU = np.array([1.3, 0.7])
ALPHA = np.array([[0.20, 0.12], [0.08, 0.16]])
BETA = np.array([[1.0, 1.5], [1.2, 0.8]])


def batch_from_arrays(times, marks):
    times = torch.as_tensor(np.asarray(times), dtype=torch.float64)
    return Batch(
        times,
        torch.diff(times, dim=1, prepend=torch.zeros(len(times), 1)),
        torch.as_tensor(np.asarray(marks), dtype=torch.long),
        torch.ones_like(times, dtype=torch.bool),
    )


def marked_reference(rng, process, size, events=6, mu=MU, alpha=ALPHA, beta=BETA):
    """Independent NumPy Ogata oracle; marks use intensity AFTER decay.

    Empty history and a fixed count of arrivals; no stationary/burn-in claim.
    alpha[k,m] is the integrated offspring contribution from mark m to k.
    """
    mu, alpha, beta = map(np.asarray, (mu, alpha, beta))
    k = len(mu)
    if (
        process not in ("poisson", "hawkes")
        or size < 1
        or events < 1
        or mu.shape != (k,)
        or alpha.shape != (k, k)
        or beta.shape != (k, k)
        or not all(np.isfinite(v).all() for v in (mu, alpha, beta))
        or np.any(mu <= 0)
        or np.any(alpha < 0)
        or np.any(beta <= 0)
    ):
        raise ValueError("Invalid marked reference parameters")
    times, marks = [], []
    for _ in range(size):
        row_t, row_m, t = [], [], 0.0
        excitation = np.zeros((k, k))
        while len(row_t) < events:
            bound = float((mu + excitation.sum(axis=1)).sum())
            dt = rng.exponential(1 / bound)
            t += dt
            excitation *= np.exp(-beta * dt)
            intensity = mu + excitation.sum(axis=1)
            if rng.random() * bound <= intensity.sum():
                mark = int(rng.choice(k, p=intensity / intensity.sum()))
                row_t.append(t)
                row_m.append(mark)
                if process == "hawkes":
                    excitation[:, mark] += alpha[:, mark] * beta[:, mark]
        times.append(row_t)
        marks.append(row_m)
    return batch_from_arrays(times, marks)


def select(batch, ids, dtype=None):
    tensors = [t[ids].clone() for t in batch.to_mapping().values()]
    if dtype:
        tensors[:2] = [t.to(dtype) for t in tensors[:2]]
    return Batch(*tensors)


def fingerprint(batch):
    digest = hashlib.sha256()
    for name, tensor in batch.to_mapping().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_model(name, train, output, seed, steps=200):
    """Small deterministic CPU experiment using production likelihoods and models."""
    import new_ltpp.models  # noqa: F401 -- populate the public registry

    torch.manual_seed(seed)
    raw = json.loads(Path("tests/integration/config.json").read_text())["model_config"]
    raw["model_id"] = name
    raw["specs"]["hidden_size"] = 8
    raw["thinning_config"].update(
        num_sample=1, num_exp=20, num_samples_boundary=32, over_sample_rate=5.0
    )
    kwargs = (
        dict(mu=[0.7, 0.3], alpha=[[0.1, 0.04], [0.02, 0.08]])
        if name == "Hawkes"
        else {}
    )
    model = ModelFactory.create_model_by_name(
        name,
        ModelConfig.model_validate(raw),
        dict(num_event_types=2, pad_token_id=2, end_time_max=20.0, dtime_max=10.0),
        output / name,
        **kwargs,
    )
    model._simulator = Simulator(model)
    train = select(train, slice(None), torch.float32)
    initial = {k: v.clone() for k, v in model.state_dict().items()}

    def training_nll():
        # The same full batch and integration RNG before/after fitting.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed + 1000)
            loss, count = model.loglike_loss(train)
            return float(loss / count)

    initial_nll = training_nll()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    model.train()
    for _ in range(steps):
        batch = select(train, torch.randperm(len(train.time_seqs))[:32])
        optimizer.zero_grad()
        loss, count = model.loglike_loss(batch)
        loss = loss / count
        if not torch.isfinite(loss):
            raise ArithmeticError("Nonfinite training likelihood")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    if not any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()):
        raise AssertionError("Model parameters did not learn")
    checkpoint = output / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    metadata = dict(
        model=name,
        seed=seed,
        steps=steps,
        learning_rate=0.01,
        training_data_sha256=fingerprint(train),
        training_sequences=len(train.time_seqs),
        first_minibatch_nll=losses[0],
        last_minibatch_nll=losses[-1],
        initial_training_nll_per_event=initial_nll,
        final_training_nll_per_event=training_nll(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        configuration=raw,
    )
    if name == "Hawkes":
        metadata["effective_parameters"] = {
            key: getattr(model, key + "_pos").detach().numpy().tolist()
            for key in ("mu", "alpha", "beta")
        }
        metadata["branching_spectral_radius"] = float(
            max(abs(np.linalg.eigvals(metadata["effective_parameters"]["alpha"])))
        )
    return model, metadata


def model_pool(model, size, events, seed):
    """Independent continuations from empty history, chunked for bounded memory."""
    torch.manual_seed(seed)
    chunks = []
    with torch.no_grad():
        for offset in range(0, size, 128):
            batch = model.simulate_from_scratch(min(128, size - offset), events)
            if (
                not torch.isfinite(batch.time_seqs).all()
                or not batch.valid_event_mask.all()
            ):
                raise ArithmeticError("Invalid learned-model simulation")
            chunks.append(batch)
    return Batch(
        *(
            torch.cat([b.to_mapping()[key] for b in chunks])
            for key in chunks[0].to_mapping()
        )
    )


def evaluate(label, hypothesis, x, y, trials, permutations, seed):
    """Raw observations permuted and pairwise truncation recomputed each time."""
    test = MMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), permutations
    )
    generator = torch.Generator().manual_seed(seed)
    p_values, order_digest = [], hashlib.sha256()
    n = 8
    for trial in range(trials):
        ids = slice(trial * n, (trial + 1) * n)
        orders = torch.stack(
            [torch.arange(2 * n)]
            + [torch.randperm(2 * n, generator=generator) for _ in range(permutations)]
        )
        order_digest.update(orders.numpy().tobytes())
        stats = paired_statistics(test, select(x, ids), select(y, ids), orders)[
            "recomputed_pairwise_truncation"
        ]
        p_values.append(permutation_p_value(stats[0], stats[1:]))
        if (trial + 1) % 50 == 0:
            print(f"{label}: {trial + 1}/{trials}", flush=True)
    rejected = sum(p <= 0.05 for p in p_values)
    result = dict(
        experiment=label,
        hypothesis=hypothesis,
        trials=trials,
        rejections=rejected,
        rejection_rate=rejected / trials,
        wilson_95=wilson_interval(rejected, trials),
        p_values=p_values,
        x_sha256=fingerprint(x),
        y_sha256=fingerprint(y),
        permutations_sha256=order_digest.hexdigest(),
    )
    print(f"{label}: {rejected}/{trials}, CI {result['wilson_95']}", flush=True)
    return result


def run(
    output, trials=300, diagnostic_trials=100, permutations=99, steps=200, seed=20261013
):
    if min(trials, diagnostic_trials, permutations, steps) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(seed).spawn(20)
    rngs = [np.random.default_rng(s) for s in streams]
    results, fits = [], []
    report = dict(
        seed=seed,
        permutations=permutations,
        alpha=0.05,
        samples_per_group=8,
        events_per_sequence=6,
        marks=2,
        protocol="pairwise_truncation_recomputed_v1",
        p_value_protocol="plus_one_float64_conservative_ties_v1",
        reference_parameters=dict(
            mu=MU.tolist(), alpha=ALPHA.tolist(), beta=BETA.tolist()
        ),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        results=results,
        fits=fits,
        status="in_progress",
        scope="Fixed event count, empty history, independent cohorts conditional on one frozen fit; no heterogeneous-history or all-model validity claim.",
    )

    def save():
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    for i, process in enumerate(("poisson", "hawkes")):
        x = marked_reference(rngs[2 * i], process, 8 * trials)
        y = marked_reference(rngs[2 * i + 1], process, 8 * trials)
        results.append(
            evaluate(
                f"marked_{process}_null", "H0", x, y, trials, permutations, seed + i
            )
        )
        x = marked_reference(rngs[4 + 2 * i], process, 8 * diagnostic_trials)
        y = marked_reference(rngs[5 + 2 * i], process, 8 * diagnostic_trials)
        # Relabeling changes the mark law without changing any generated timestamp.
        y.type_seqs = 1 - y.type_seqs
        results.append(
            evaluate(
                f"marked_{process}_mark_swap",
                "H1_marks",
                x,
                y,
                diagnostic_trials,
                permutations,
                seed + 2 + i,
            )
        )
        save()
    train = marked_reference(rngs[8], "hawkes", 256)
    for i, name in enumerate(("Hawkes", "NHP")):
        model, metadata = fit_model(name, train, output, seed + 10 + i, steps)
        fits.append(metadata)
        pool = model_pool(model, 16 * trials, 6, seed + 20 + i)
        x, y = select(pool, slice(0, 8 * trials)), select(pool, slice(8 * trials, None))
        results.append(
            evaluate(
                f"learned_{name}_internal_null",
                "H0_conditional_on_fit",
                x,
                y,
                trials,
                permutations,
                seed + 30 + i,
            )
        )
        if name == "Hawkes":
            oracle = marked_reference(
                rngs[9], "hawkes", 8 * trials, **metadata["effective_parameters"]
            )
            results.append(
                evaluate(
                    "learned_Hawkes_independent_oracle",
                    "H0_conditional_on_fit",
                    x,
                    oracle,
                    trials,
                    permutations,
                    seed + 40,
                )
            )
        heldout = marked_reference(rngs[10 + i], "hawkes", 8 * diagnostic_trials)
        generated = model_pool(model, 8 * diagnostic_trials, 6, seed + 50 + i)
        results.append(
            evaluate(
                f"learned_{name}_target_fit",
                "fit_diagnostic_not_type_I",
                heldout,
                generated,
                diagnostic_trials,
                permutations,
                seed + 60 + i,
            )
        )
        save()
    report["status"] = "completed"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/extended-scientific")
    )
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--diagnostic-trials", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
