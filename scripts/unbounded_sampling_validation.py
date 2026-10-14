"""Audit sampler changes without silently replacing historical references."""

import argparse
import json
from pathlib import Path

from new_ltpp.models.event_sampler import SamplingSafetyError
from scripts.integration_suite import (
    CATALOGUE,
    FIXTURES,
    compare,
    compare_reference,
    run_model,
    write_json,
)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    previous = json.loads(
        Path(
            "docs/validation/parametric-refactor-20261014/architecture.json"
        ).read_text()
    )
    for name in CATALOGUE:
        target = output / name
        target.mkdir(parents=True, exist_ok=True)
        try:
            actual = run_model(
                name, target, sampling_rounds=128 if name == "FullyNN" else None
            )
        except RuntimeError as error:
            if name != "FullyNN":
                raise
            cause = error
            while cause is not None and not isinstance(cause, SamplingSafetyError):
                cause = cause.__cause__
            if cause is None:
                raise
            manifests = list(target.rglob("manifest.json"))
            assert len(manifests) == 1
            manifest = json.loads(manifests[0].read_text())
            assert any(p["status"] == "failed" for p in manifest["phases"].values())
            assert not (target / "snapshot.json").exists()
            report[name] = dict(
                status="known_model_limitation_explicit_safety_stop",
                reason=cause.reason,
                rounds=cause.rounds,
                pending=cause.pending_count,
                fabricated_events=False,
                phases=manifest["phases"],
                code=manifest["code"],
            )
        else:
            if name == "FullyNN":
                raise AssertionError(
                    "Review FullyNN: deterministic fixture unexpectedly completed despite its bounded hazard"
                )
            if name == "Poisson":
                report[name] = dict(
                    status="passed_new_parametric_pipeline_unchanged_algorithm"
                )
            else:
                expected = json.loads(
                    (FIXTURES / "baselines" / f"{name}.json").read_text()
                )
                if name in ("Hawkes", "SelfCorrecting"):
                    for field, delta in previous[name]["changes"].items():
                        expected[field] = delta["after"]
                    compare(actual, expected)
                    report[name] = dict(status="passed_parametric_reference_unchanged")
                else:
                    pure = (
                        "model",
                        "config_sha256",
                        "fixture_sha256",
                        "trained_state",
                        "probe_loss",
                        "probe_num_events",
                        "probe_intensities",
                        "probe_gradients",
                    )
                    compare_reference(
                        {k: actual[k] for k in pure}, {k: expected[k] for k in pure}
                    )
                    changes = {
                        field: dict(before=expected[field], after=actual[field])
                        for field in (
                            "test_metrics",
                            "simulation_numeric_results",
                            "simulation_records",
                        )
                        if actual[field] != expected[field]
                    }
                    report[name] = dict(
                        status="passed_neural_pipeline_intended_sampling_changes",
                        likelihood_and_intensity_calculations_unchanged=True,
                        changes=changes,
                    )
        write_json(output / "report.json", report)
        print(name, report[name]["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
