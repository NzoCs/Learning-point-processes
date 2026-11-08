"""Train actual library models and test their independent conditional generations."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_rows(rows, scale, marks, dataset, split):
    output = []
    for i, row in enumerate(rows):
        times = np.asarray(row["time_since_start"], dtype=float)
        types = np.asarray(row["type_event"])
        if (
            len(times) != len(types)
            or len(times) < 1
            or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0)
            or times[0] < 0
        ):
            raise ValueError("Invalid training trajectory; no records filtered")
        if (
            not np.isfinite(types).all()
            or np.any(types != types.astype(int))
            or np.any((types < 0) | (types >= marks))
        ):
            raise ValueError("Invalid marks")
        values = times / scale
        output.append(
            dict(
                dim_process=marks,
                sequence_id=f"{dataset}:{split}:{i}",
                time_since_start=values.tolist(),
                time_since_last_event=np.diff(values, prepend=0.0).tolist(),
                type_event=types.astype(int).tolist(),
            )
        )
    return output


def prepare(root, plan, cache):
    import yaml
    from datasets import load_dataset

    target = root / "data"
    target.mkdir()
    specs = yaml.safe_load(Path("yaml_configs/configs.yaml").read_text())[
        "data_configs"
    ]
    for name in plan["datasets"]:
        cfg = specs[name]
        ds = load_dataset(
            cfg["src_dir"], revision=cfg["revision"], cache_dir=str(cache)
        )
        training = ds["train"]
        positive = np.concatenate(
            [np.diff(np.asarray(r["time_since_start"], dtype=float)) for r in training]
        )
        positive = positive[positive > 0]
        scale = float(8 * np.median(positive))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Training scale undefined")
        directory = target / name
        directory.mkdir()
        summary = dict(
            repo=cfg["src_dir"],
            revision=cfg["revision"],
            time_divisor=scale,
            num_event_types=cfg["num_event_types"],
            splits={},
            context=None,
        )
        for source, label in [
            ("train", "train"),
            ("validation", "dev"),
            ("test", "test"),
        ]:
            actual = (
                source if source in ds else "dev" if source == "validation" else source
            )
            rows = normalize_rows(
                ds[actual], scale, cfg["num_event_types"], name, label
            )
            path = directory / (label + ".json")
            with path.open("x") as f:
                for row in rows:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            summary["splits"][label] = dict(
                n=len(rows),
                path=str(path),
                sha256=sha(path),
                fingerprint=ds[actual]._fingerprint,
                zero_gap_count=sum(
                    sum(v == 0 for v in r["time_since_last_event"][1:]) for r in rows
                ),
            )
            if label == "train":
                for row in rows:
                    count = plan["history_events"]
                    t = row["time_since_start"][:count]
                    if len(t) == count and np.all(np.diff(t) > 0):
                        summary["context"] = dict(
                            times=t,
                            marks=row["type_event"][:count],
                            source_id=row["sequence_id"],
                        )
                        break
        if summary["context"] is None:
            raise ValueError("No eligible training-only conditioning history")
        summary["interpretation"] = (
            "Published splits; independent physical units and terminal exposure not certified. Native held-out metrics are descriptive; synthetic model generations have independent RNG streams."
        )
        write(directory / "manifest.json", summary)
    write(root / "data-ready.json", dict(status="complete", datasets=plan["datasets"]))


def cell_spec(plan, index):
    return (
        plan["models"][index // len(plan["datasets"])],
        plan["datasets"][index % len(plan["datasets"])],
    )


def observed_reference_order(pools, receipts, alternative):
    """The alternative is stream1; Stein must score it as observed, against Q."""
    return (pools[::-1], receipts[::-1]) if alternative else (pools, receipts)


def fit(root, plan, index):
    import new_ltpp.models  # noqa: F401
    from new_ltpp.configs import RunnerConfig
    from new_ltpp.configs.model_config import ThinningConfig
    from new_ltpp.runners.runner_manager import RunnerManager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU fallback")
    model, dataset = cell_spec(plan, index)
    cell = root / "cells" / str(index)
    cell.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "data" / dataset / "manifest.json").read_text())
    paths = dict(
        data_config_path="data_configs." + dataset,
        general_specs_config_path="general_specs.h16",
        training_config_path="training_configs.quick_test",
        data_loading_config_path="data_loading_configs.b32_w1",
        thinning_config_path="thinning_configs.quick_test",
        simulation_config_path="simulation_configs.quick_test",
        statistical_test_config_path="statistical_test_configs.sig_kernel",
    )
    config = RunnerConfig.from_yaml_presets(
        "yaml_configs/configs.yaml",
        paths,
        model,
        max_epochs=plan["epochs_cap"],
        save_dir=str(cell / "training"),
        run_id=f'trained-{model}-{dataset}-{plan["training_seed"]}',
    )
    config.data_config = config.data_config.model_copy(
        update=dict(
            data_format="json",
            revision=None,
            train_dir=manifest["splits"]["train"]["path"],
            valid_dir=manifest["splits"]["dev"]["path"],
            test_dir=manifest["splits"]["test"]["path"],
        )
    )
    config.training_config = config.training_config.model_copy(
        update=dict(
            seed=plan["training_seed"],
            devices=1,
            deterministic=True,
            val_freq=1,
            checkpoints_freq=1,
            patience=plan["patience"],
            use_precision_16=False,
        )
    )
    config.model_cfg = config.model_cfg.model_copy(
        update=dict(
            device="cuda",
            gpu=0,
            thinning_config=ThinningConfig(
                num_exp=32,
                num_sample=16,
                num_samples_boundary=32,
                over_sample_rate=5.0,
                max_sampling_rounds=10000,
            ),
        )
    )
    write(cell / "effective-config.json", config.get_yaml_config())
    manager = RunnerManager(config)
    try:
        manager.run("train")
        checkpoint = Path(manager.runner.checkpoint_path)
        if not checkpoint.is_file():
            raise RuntimeError("Validation-selected checkpoint missing")
        report = dict(
            status="trained",
            model=model,
            dataset=dataset,
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            selection="Lowest native validation loss; evaluation p-values never select weights",
            native_data_manifest=sha(root / "data" / dataset / "manifest.json"),
        )
        try:
            manager.run("test")
            report["native_heldout_status"] = "complete"
        except Exception as error:
            report["native_heldout_status"] = "blocked"
            report["native_heldout_error"] = dict(
                type=type(error).__name__, message=str(error)
            )
        write(cell / "fit.json", report)
    except Exception as error:
        write(
            cell / "fit.json",
            dict(
                status="failed",
                model=model,
                dataset=dataset,
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise


def fixed_window(model, context, n, horizon, seed, initial=32, maximum=4096):
    from new_ltpp.models.simulation.simulator import Simulator
    from new_ltpp.shared_types import Batch

    parameter = next(model.parameters())
    device = parameter.device
    times = torch.tensor(context["times"], dtype=parameter.dtype, device=device)[
        None
    ].repeat(n, 1)
    marks = torch.tensor(context["marks"], dtype=torch.long, device=device)[
        None
    ].repeat(n, 1)
    history = Batch(
        times,
        torch.diff(times, dim=1, prepend=times.new_zeros(n, 1)),
        marks,
        torch.ones_like(marks, dtype=torch.bool),
    )
    engine = Simulator(model)
    budget = initial
    while True:
        torch.manual_seed(seed)
        data = engine.simulate(history, num_events_to_simulate=budget)
        mask = data.valid_event_mask
        if any(
            not torch.isfinite(t[m]).all()
            or not (t[m] > 0).all()
            or not (torch.diff(t[m]) > 0).all()
            for t, m in zip(data.time_seqs, mask)
        ):
            raise RuntimeError(
                "Invalid finite generated times; no jitter or exclusions"
            )
        end = data.time_seqs.masked_fill(~mask, -torch.inf).max(1).values
        terminal = data.no_next_event_mask
        terminal = (
            torch.zeros(n, dtype=torch.bool, device=device)
            if terminal is None
            else terminal
        )
        if ((end >= horizon) | terminal).all():
            observed = mask & (data.time_seqs < horizon)
            out = Batch(
                data.time_seqs.masked_fill(~observed, 0),
                data.time_delta_seqs.masked_fill(~observed, 0),
                data.type_seqs.masked_fill(~observed, 0),
                observed,
            )
            return out, dict(
                event_budget=budget,
                coverage_end=horizon,
                crossing_events=int((end >= horizon).sum()),
                certified_no_next_event=int(terminal.sum()),
            )
        if budget >= maximum:
            raise RuntimeError(
                "Event budget did not cover horizon; tail is not assumed empty"
            )
        budget = min(2 * budget, maximum)


def benchmark(root, plan, index):
    import new_ltpp.models  # noqa: F401
    from new_ltpp.configs import ModelConfig
    from new_ltpp.configs.window_test_config import WindowTestConfig
    from new_ltpp.evaluation.statistical_testing.references import (
        NeuralIntensityReference,
    )
    from new_ltpp.models.model_factory import ModelFactory
    from scripts.ruche_trained_test_pilot import evaluate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required, no CPU fallback")
    cell = root / "cells" / str(index)
    output = cell / "benchmark"
    output.mkdir(exist_ok=False)
    fitted = json.loads((cell / "fit.json").read_text())
    if (
        fitted["status"] != "trained"
        or sha(fitted["checkpoint"]) != fitted["checkpoint_sha256"]
    ):
        raise RuntimeError("Valid unchanged trained checkpoint required")
    _, dataset = cell_spec(plan, index)
    manifest = json.loads((root / "data" / dataset / "manifest.json").read_text())
    context = manifest["context"]
    checkpoint = torch.load(
        fitted["checkpoint"], map_location="cpu", weights_only=False
    )
    model = ModelFactory.create_model_by_name(
        fitted["model"],
        ModelConfig.model_validate(checkpoint["hyper_parameters"]["model_config"]),
        checkpoint["hyper_parameters"]["data_info"],
        output / "model",
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()
    reference = NeuralIntensityReference(
        model, history_times=context["times"], history_marks=context["marks"]
    )
    settings = [
        WindowTestConfig(
            horizon=1,
            num_event_types=model.num_event_types,
            seed=plan["evaluation_seed"],
            **m,
        )
        for m in plan["methods"]
    ]

    def pair(n, stage, trial, hypothesis):
        pools = []
        receipts = []
        for group in [0, 1]:
            compression = plan["time_compression"] if hypothesis and group == 1 else 1.0
            seed = int(
                np.random.SeedSequence(
                    [plan["evaluation_seed"], index, stage, trial, group]
                ).generate_state(1)[0]
            )
            started = time.perf_counter()
            data, receipt = fixed_window(
                model,
                context,
                n,
                1 / compression,
                seed,
                plan["initial_event_budget"],
                plan["maximum_event_budget"],
            )
            data.time_seqs *= compression
            data.time_delta_seqs *= compression
            receipt["elapsed_seconds"] = time.perf_counter() - started
            pools.append(data)
            receipts.append(receipt)
        return observed_reference_order(pools, receipts, hypothesis)

    try:
        availability = []
        for method, cfg in enumerate(settings):
            try:
                for n in [16, 128]:
                    pools, receipts = pair(n, 0, 0, 0)
                    report = evaluate(
                        *pools, reference, [cfg], plan["cost_gate_seconds_per_method"]
                    )
                    report["generation_coverage"] = receipts
                    write(output / f"cost-{method}-{n}.json", report)
                availability.append(dict(method=method, status="supported"))
            except Exception as error:
                availability.append(
                    dict(
                        method=method,
                        status="cost_or_capability_blocked",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                )
        write(output / "availability.json", availability)
        active = [i for i, a in enumerate(availability) if a["status"] == "supported"]
        if not active:
            raise RuntimeError("No test passed capability/cost gates")
        for hyp, label in enumerate(["h0", "time_compression"]):
            with (output / (label + "-trials.jsonl")).open("x") as stream:
                for trial in range(plan["trials_per_hypothesis"]):
                    pools, receipts = pair(plan["n"], 1, trial, hyp)
                    for group, data in enumerate(pools):
                        np.savez_compressed(
                            output / f"{label}-{trial:03d}-g{group}.npz",
                            times=data.time_seqs.cpu().numpy(),
                            marks=data.type_seqs.cpu().numpy(),
                            mask=data.valid_event_mask.cpu().numpy(),
                            horizon=np.array(1.0),
                        )
                    tests = []
                    for i in active:
                        seed = int(
                            np.random.SeedSequence(
                                [plan["evaluation_seed"], index, 1, trial, 2]
                            ).generate_state(1)[0]
                        )
                        cfg = settings[i].model_copy(update={"seed": seed})
                        result = evaluate(
                            *pools,
                            reference,
                            [cfg],
                            plan["cost_gate_seconds_per_method"],
                        )
                        result["method_index"] = i
                        tests.append(result)
                    row = dict(
                        trial=trial,
                        hypothesis=label,
                        model=fitted["model"],
                        dataset=dataset,
                        generation_coverage=receipts,
                        methods=tests,
                    )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
        write(
            output / "summary.json",
            dict(
                status="complete_supported_substudies",
                checkpoint_sha256=fitted["checkpoint_sha256"],
                availability=availability,
                n=plan["n"],
                trials_per_hypothesis=plan["trials_per_hypothesis"],
                full_suite_complete=len(active) == len(settings),
                scope=plan["scope"],
            ),
        )
    except Exception as error:
        write(
            output / "summary.json",
            dict(status="failed", error_type=type(error).__name__, error=str(error)),
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=["prepare", "fit", "benchmark"], required=True)
    p.add_argument("--index", type=int, choices=range(4))
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    torch.set_num_threads(1)
    if a.phase == "prepare":
        prepare(a.root, plan, a.cache)
    elif a.phase == "fit":
        fit(a.root, plan, a.index)
    else:
        benchmark(a.root, plan, a.index)
