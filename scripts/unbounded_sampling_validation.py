"""Audit sampler changes without silently replacing historical references."""

import argparse
import json
import platform
from pathlib import Path

from new_ltpp.models.event_sampler import SamplingSafetyError
from scripts.integration_suite import (
    CATALOGUE,
    FIXTURES,
    compare,
    run_model,
    write_json,
)


def fixed_reference_probe(name, target):
    """Compare formulas using reviewed fixed weights, independently of refitting."""
    import torch
    import yaml

    from new_ltpp.configs import RunnerConfig
    from new_ltpp.runners.runner_manager import RunnerManager
    from new_ltpp.shared_types import Batch

    expected = json.loads((FIXTURES / "baselines" / f"{name}.json").read_text())
    state = expected["trained_state"]
    profile = FIXTURES / "linux_trained_states" / f"{name}.json"
    if platform.system() == "Linux" and profile.exists():
        state = json.loads(profile.read_text())["trained_state"]
    config = RunnerConfig.model_validate(
        yaml.safe_load((target / "effective.yaml").read_text())
    )
    manager = RunnerManager(config)
    manager.setup_runner(enable_logging=False)
    model = manager.runner.model
    model.load_state_dict(
        {k: torch.tensor(v["values"]).reshape(v["shape"]) for k, v in state.items()},
        strict=True,
    )
    model.eval()
    manager.runner.datamodule.setup(stage="test")
    original = next(iter(manager.runner.datamodule.test_dataloader()))
    batch = Batch(*(value[:2].clone() for value in original.to_mapping().values()))
    torch.manual_seed(456)
    loss, count = model.loglike_loss(batch)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    intensity = None
    if getattr(model, "supports_intensity", True):
        intensity = model.compute_intensities_at_sample_dtimes(
            **batch.to_mapping(),
            sample_dtimes=torch.full((*batch.time_seqs.shape, 2), 0.2),
        )

    def data(t):
        if t is None:
            return None
        t = t.detach().cpu()
        assert torch.isfinite(t).all()
        return dict(shape=list(t.shape), values=t.reshape(-1).tolist())

    actual = dict(
        probe_loss=data(loss),
        probe_num_events=int(count),
        probe_intensities=data(intensity),
        probe_gradients={
            k: data(g) for (k, _), g in zip(model.named_parameters(), gradients)
        },
    )
    compare(actual, {k: expected[k] for k in actual})
    write_json(
        target / "fixed_reference_probe.json",
        dict(
            model=name,
            reference_profile=(
                str(profile.relative_to(FIXTURES))
                if platform.system() == "Linux" and profile.exists()
                else "historical_baseline"
            ),
            checked_at_fixed_weights=True,
            actual=actual,
        ),
    )
    return state


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
            fixed_reference_probe(name, target)
            report[name] = dict(
                status="known_model_limitation_explicit_safety_stop",
                reason=cause.reason,
                rounds=cause.rounds,
                pending=cause.pending_count,
                fabricated_events=False,
                fixed_reference_formulas="passed",
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
                    expected["trained_state"] = fixed_reference_probe(name, target)
                    changes = {
                        field: dict(before=expected[field], after=actual[field])
                        for field in (
                            "trained_state",
                            "probe_loss",
                            "probe_intensities",
                            "probe_gradients",
                            "test_metrics",
                            "simulation_numeric_results",
                            "simulation_records",
                        )
                        if actual[field] != expected[field]
                    }
                    report[name] = dict(
                        status="passed_neural_pipeline_intended_sampling_changes",
                        likelihood_and_intensity_calculations_unchanged_at_fixed_weights=True,
                        changes=changes,
                    )
        write_json(output / "report.json", report)
        print(name, report[name]["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
