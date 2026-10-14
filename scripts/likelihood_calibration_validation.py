"""Two independent-fit checkpoints, stopping and window calibration, no Ruche."""

import argparse
import json
from pathlib import Path

from scripts.conditional_integration_suite import run_model
from scripts.fixed_window_calibration import run as window_calibration
from scripts.integration_suite import write_json
from scripts.stopped_conditional_calibration import run as stopped_calibration


def run(output, trials=300, power_trials=100):
    report = {}
    for name in ("FullyNN", "RMTPP"):
        for epochs in (1, 2):
            target = output / f"{name}-epochs-{epochs}"
            run_model(name, target / "fit", epochs=epochs)
            manifests = list((target / "fit").rglob("manifest.json"))
            assert len(manifests) == 1
            manifest = json.loads(manifests[0].read_text())
            checkpoint = Path(manifest["phases"]["train"]["checkpoint"]["path"])
            # 'last' distinguishes epochs even if the best checkpoint is epoch 1.
            checkpoint = checkpoint.parent / "last.ckpt"
            import torch

            from scripts.integration_suite import tensor_data

            fitted_state = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )["state_dict"]
            write_json(
                target / "frozen_parameters.json",
                {key: tensor_data(value) for key, value in fitted_state.items()},
            )
            stopped = stopped_calibration(
                name,
                target / "stopped",
                checkpoint=checkpoint,
                trials=trials,
                power_trials=power_trials,
                seed=20261102 + epochs * 100,
                h0_family_size=16,
            )
            window = window_calibration(
                name,
                target / "window",
                checkpoint=checkpoint,
                trials=trials,
                power_trials=power_trials,
                seed=20261103 + epochs * 100,
            )
            # Checkpoint parameters are frozen in both experiments. H0 decisions
            # share a predeclared family threshold over all 16 comparisons.
            evidence = []
            for design, result in (("stopped", stopped), ("window", window)):
                for row in result["results"]:
                    if (
                        row["hypothesis"] == "H0"
                        and row.get("protocol", "complete_stopped")
                        == "complete_stopped"
                    ):
                        evidence.append(
                            dict(
                                design=design,
                                method=row["method"],
                                rate=row["rejection_rate"],
                                interval=row["wilson_95"],
                                excess_detected=bool(
                                    row["excess_binomial_p"] < 0.01 / 16
                                ),
                            )
                        )
            report[f"{name}-{epochs}"] = dict(
                checkpoint=str(checkpoint),
                checkpoint_sha256=window["fit"]["sha256"],
                model_state_sha256=window["fit"]["model_state_sha256"],
                h0=evidence,
                completed=True,
                no_h0_excess_detected=not any(r["excess_detected"] for r in evidence),
            )
            write_json(output / "report.json", report)
    for name in ("FullyNN", "RMTPP"):
        assert (
            report[f"{name}-1"]["checkpoint_sha256"]
            != report[f"{name}-2"]["checkpoint_sha256"]
        )
        assert (
            report[f"{name}-1"]["model_state_sha256"]
            != report[f"{name}-2"]["model_state_sha256"]
        )
    if any(not r["no_h0_excess_detected"] for r in report.values()):
        raise AssertionError("Predeclared H0 excess detected; raw results retained")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--power-trials", type=int, default=100)
    run(**vars(parser.parse_args()))
