"""Conditional H0/H1 experiments for each frozen learned model; oracle and internal laws explicit."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

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
from scripts.conditional_calibration import (
    cached_paired_statistics,
    conditional_hawkes_reference,
    heterogeneous_histories,
    swap_orders,
)
from scripts.extended_scientific_validation import (
    code_provenance,
    fingerprint,
    fit_model,
    marked_reference,
)
from scripts.integration_suite import MODELS, write_json
from scripts.scientific_calibration import wilson_interval


def alternative(batch, histories, kind):
    """Predeclared changes to heldout continuations; preserve history and origin."""
    from scripts.extended_scientific_validation import select

    changed = select(batch, slice(None))
    if kind == "time_scale_x4":
        changed.time_seqs *= 4
        changed.time_delta_seqs *= 4
    elif kind == "context_marks":
        changed.type_seqs = (
            histories.type_seqs[:, -1:].expand_as(changed.type_seqs).clone()
        )
    else:
        raise ValueError(kind)
    return changed


def run(
    name,
    output,
    trials=500,
    power_trials=200,
    permutations=99,
    steps=200,
    seed=20261014,
):
    if min(trials, power_trials, permutations, steps) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    training = marked_reference(np.random.default_rng(seed), "hawkes", 256)
    model, fit = fit_model(name, training, output, seed + 1, steps)
    frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
    test = ConditionalMMDTwoSampleTest(
        SIGKernel(RBFKernel(), "counting_grid", 8, 0, 2), permutations
    )
    report = dict(
        status="in_progress",
        model=name,
        seed=seed,
        code=code_provenance(__file__),
        fit=fit,
        trials_H0=trials,
        trials_H1_per_alternative=power_trials,
        permutations=permutations,
        batch_sizes=[8, 5, 3],
        events_per_continuation=6,
        marks=2,
        history_events=3,
        alpha=0.05,
        null_method="within_history_pair_swaps_joint_kernel_v1",
        results=[],
        observation_law=(
            "independent_numpy_hawkes_oracle_at_fitted_parameters"
            if name == "Hawkes"
            else "internal_simulator_conditional_law"
        ),
        scope="fixed independently fitted checkpoint; internal H0 does not validate the sampler against an external law or composite fit uncertainty",
    )
    write_json(output / "report.json", report)
    for case_id, case in enumerate(("H0", "time_scale_x4", "context_marks")):
        count = trials if case == "H0" else power_trials
        values = dict(single_first_batch=[], aggregated=[])
        data_hash, orders_hash = hashlib.sha256(), hashlib.sha256()
        for trial in range(count):
            test.reset_accumulators()
            for block, n in enumerate(report["batch_sizes"]):
                histories = heterogeneous_histories(n)
                sample_seed = seed + 1000000 * case_id + 1000 * trial + 10 * block
                torch.manual_seed(sample_seed + 10000000)
                with torch.no_grad():
                    simulated = model.simulate(histories, num_events_to_simulate=6)
                if name == "Hawkes":
                    observed = conditional_hawkes_reference(
                        np.random.default_rng(sample_seed + 20000000),
                        histories,
                        **fit["effective_parameters"],
                    )
                else:
                    torch.manual_seed(sample_seed + 20000000)
                    with torch.no_grad():
                        observed = model.simulate(histories, num_events_to_simulate=6)
                if case != "H0":
                    observed = alternative(observed, histories, case)
                order_seed = sample_seed + 30000000
                orders = swap_orders(n, permutations, order_seed)
                stats = cached_paired_statistics(
                    test, histories, observed, simulated, orders
                )
                if trial == 0:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(order_seed)
                        actual = test.compute_statistics(
                            observed,
                            simulated,
                            histories=histories,
                            pair_ids=[f"{case}/{trial}/{block}/{i}" for i in range(n)],
                            accumulate=False,
                        )
                    torch.testing.assert_close(
                        stats[0], actual["observed_statistic"], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        stats[1:], actual["permuted_statistics"], rtol=0, atol=0
                    )
                assert torch.isfinite(stats).all()
                test._accumulate(stats[0], stats[1:])
                if block == 0:
                    values["single_first_batch"].append(
                        float(monte_carlo_p_value(stats[0], stats[1:]))
                    )
                data_hash.update(
                    (
                        fingerprint(histories)
                        + fingerprint(observed)
                        + fingerprint(simulated)
                    ).encode()
                )
                orders_hash.update(orders.numpy().tobytes())
            values["aggregated"].append(float(test.get_final_p_value()))
            if (trial + 1) % 50 == 0:
                print(f"{name}/{case}: {trial + 1}/{count}", flush=True)
        for method, p_values in values.items():
            rejections = sum(p <= 0.05 for p in p_values)
            result = dict(
                hypothesis=case,
                method=method,
                trials=count,
                rejections=rejections,
                rejection_rate=rejections / count,
                wilson_95=wilson_interval(rejections, count),
                p_values=p_values,
                raw_data_sha256=data_hash.hexdigest(),
                permutations_sha256=orders_hash.hexdigest(),
                public_api_all_draws_checked_on_first_trial=True,
            )
            report["results"].append(result)
            print(
                f"{name}/{case}/{method}: {rejections}/{count}, CI {result['wilson_95']}",
                flush=True,
            )
        write_json(output / "report.json", report)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    report["model_frozen_verified"] = True
    report["status"] = "completed"
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--power-trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261014)
    args = vars(parser.parse_args())
    args["name"] = args.pop("model")
    run(**args)


if __name__ == "__main__":
    main()
