"""Audit all production pipelines against the last reviewed pre-wiring snapshots."""

import argparse
import json
from pathlib import Path

from scripts.conditional_integration_suite import run_model as run_conditional
from scripts.integration_suite import CATALOGUE, compare, run_model, write_json


def run(output, reference_dir=None, allow_rmtpp_likelihood_change=False):
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path("docs/validation/unbounded-sampling-20261014")
    report = {}
    unchanged = (
        "trained_state",
        "probe_loss",
        "probe_num_events",
        "probe_intensities",
        "probe_gradients",
        "test_metrics",
        "simulation_records",
        "simulation_sequence_status",
    )
    for name in CATALOGUE:
        target = output / "ordinary" / name
        target.mkdir(parents=True, exist_ok=True)
        complete = name in ("FullyNN", "RMTPP")
        actual = run_model(name, target, allow_terminated=complete)
        reference = (
            reference_dir / name / "snapshot.json"
            if reference_dir is not None
            else baseline / (name + "_snapshot.json")
        )
        before = json.loads(reference.read_text())
        if name == "RMTPP" and allow_rmtpp_likelihood_change:
            assert actual["probe_num_events"] == before["probe_num_events"]
            assert actual["fixture_sha256"] == before["fixture_sha256"]
            assert actual["config_sha256"] == before["config_sha256"]
            assert actual["statistical_protocol"] == before["statistical_protocol"]
            report[name] = dict(
                status="passed_intentional_analytic_likelihood_change",
                reason="Exact capped-exponential training compensator replaces Monte Carlo; same architecture/intensity, changed gradients and training RNG stream",
                before_snapshot=str(reference),
                after_snapshot=str(target / "snapshot.json"),
                changes={
                    field: dict(before=before[field], after=actual[field])
                    for field in unchanged
                    if before[field] != actual[field]
                },
            )
        elif complete:
            for field in unchanged:
                compare(actual[field], before[field], field)
            assert (
                actual["statistical_protocol"]["observation"]
                == "complete_stopped_event_budget_v1"
            )
            report[name] = dict(
                status="passed",
                models_predictions_and_trajectories_unchanged=True,
                changes={
                    field: dict(before=before[field], after=actual[field])
                    for field in ("statistical_protocol", "simulation_numeric_results")
                    if before[field] != actual[field]
                },
            )
        else:
            compare(actual, before)
            report[name] = dict(status="passed_all_numerical_fields_unchanged")
        conditional = output / "conditional" / name
        conditional.mkdir(parents=True, exist_ok=True)
        run_conditional(name, conditional)
        checks = json.loads((conditional / "checks.json").read_text())
        report[name]["conditional"] = checks
        if complete:
            terminal = output / "known-terminal" / name
            terminal.mkdir(parents=True, exist_ok=True)
            run_conditional(name, terminal, terminal_fixture=True)
            report[name]["known_terminal"] = json.loads(
                (terminal / "checks.json").read_text()
            )
        write_json(output / "report.json", report)
        print(name, report[name]["status"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--allow-rmtpp-likelihood-change", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), args.reference_dir, args.allow_rmtpp_likelihood_change)
