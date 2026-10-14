"""Preserve neural historical references and explicitly audit changed parametric results."""

import json
from pathlib import Path

from scripts.conditional_integration_suite import run_model as conditional
from scripts.integration_suite import (
    FIXTURES,
    MODELS,
    compare_reference,
    run_model,
    write_json,
)


def run(output):
    report = {}
    for name in MODELS:
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        actual = run_model(name, target)
        expected = json.loads((FIXTURES / "baselines" / f"{name}.json").read_text())
        if name not in ("Hawkes", "SelfCorrecting"):
            compare_reference(actual, expected)
            report[name] = dict(status="passed_historical_reference_unchanged")
        else:
            changes = {}
            for field in (
                "trained_state",
                "probe_loss",
                "probe_intensities",
                "probe_gradients",
                "test_metrics",
                "simulation_records",
                "simulation_numeric_results",
            ):
                if actual[field] != expected[field]:
                    changes[field] = dict(before=expected[field], after=actual[field])
            report[name] = dict(
                status="passed_pipeline_expected_semantic_changes",
                reason=(
                    "Removed unused embedding/RNG draws; dedicated prefix forecasts and native simulation; SelfCorrecting post-event hazard and stable inversion/integral corrected"
                    if name == "SelfCorrecting"
                    else "Removed unused embedding/RNG draws; dedicated prefix forecasts and native Ogata instead of bounded generic thinning; positive-delay intensity queries include the conditioning event"
                ),
                changes=changes,
            )
        write_json(output / "report.json", report)
        print(name, report[name]["status"], flush=True)
    for name in (*MODELS, "Poisson"):
        conditional(name, output / "conditional" / name)
        report.setdefault(name, {})["conditional_pipeline"] = "passed"
        write_json(output / "report.json", report)
        print(name, "conditional pipeline passed", flush=True)
    report["Poisson"][
        "status"
    ] = "new_parametric_model_conditional_train_test_predict_passed"
    write_json(output / "report.json", report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/architecture-regression")
    )
    run(parser.parse_args().output.resolve())
