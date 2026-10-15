"""Paired same-process replay of the original and corrected origin validators."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from new_ltpp.evaluation.statistical_testing.statistical_tests import (
    ConditionalMMDTwoSampleTest,
)
from scripts.extended_scientific_validation import fingerprint
from scripts.pipeline_protocol_calibration import (
    evaluate,
    histories_for_trial,
    model_and_config,
    observed_future,
    state_hash,
)

PARENT = "3cb3e9917896d2984675a4102d805ba4295d233e"
SOURCE = (
    "new_ltpp/evaluation/statistical_testing/statistical_tests/conditional_mmd_test.py"
)


def run(output):
    torch.set_num_threads(1)
    original = subprocess.check_output(["git", "show", f"{PARENT}:{SOURCE}"], text=True)
    cls = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.ClassDef) and n.name == "ConditionalMMDTwoSampleTest"
    )
    current = next(
        n
        for n in ast.parse(Path(SOURCE).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == cls.name
    )
    # All statistical methods in this module must be byte-equivalent ASTs.
    old_methods = {
        n.name: ast.dump(n)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name != "validate_pairs"
    }
    new_methods = {
        n.name: ast.dump(n)
        for n in current.body
        if isinstance(n, ast.FunctionDef) and n.name != "validate_pairs"
    }
    if old_methods != new_methods:
        raise AssertionError("Statistical methods changed beyond the validation fix")
    untouched = [
        "new_ltpp/models",
        "new_ltpp/evaluation/accumulators",
        "new_ltpp/evaluation/statistical_testing/point_process_kernels",
        "new_ltpp/evaluation/statistical_testing/statistical_tests/global_conditional_mmd_test.py",
        "new_ltpp/evaluation/statistical_testing/statistical_tests/mmd_test.py",
    ]
    if subprocess.check_output(
        ["git", "diff", PARENT, "HEAD", "--", *untouched]
    ).strip():
        raise AssertionError(
            "Model, sampler, kernel or global MMD implementation changed"
        )
    node = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_pairs"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            "<original-origin-validation>",
            "exec",
        ),
        namespace,
    )
    model, config, fit = model_and_config("Hawkes", output / "parent-paired")
    frozen = state_hash(model)
    rows = []
    for case_id, case in enumerate(("H0", "twice_speed", "context_marks")):
        seed = 2336101523 + case_id * 1000000
        h = histories_for_trial(128, np.random.default_rng(seed), 0)
        x = observed_future(model, h, np.random.default_rng(seed + 101), 6, seed + 201)
        if case == "twice_speed":
            x.time_seqs *= 0.5
            x.time_delta_seqs *= 0.5
        elif case == "context_marks":
            x.type_seqs = torch.where(
                x.valid_event_mask,
                h.type_seqs[:, -1:].expand_as(x.type_seqs),
                x.type_seqs,
            )
        with patch.object(
            ConditionalMMDTwoSampleTest, "validate_pairs", namespace["validate_pairs"]
        ):
            old, old_y, _ = evaluate(model, config, h, x, (64, 63, 1), 6, seed + 301)
        new, new_y, _ = evaluate(model, config, h, x, (64, 63, 1), 6, seed + 301)
        same_samples = [fingerprint(b) for b in old_y] == [
            fingerprint(b) for b in new_y
        ]
        if old != new or not same_samples:
            raise AssertionError(
                "Same-process parent/fix replay changed statistics or trajectories"
            )
        rows.append(
            dict(
                case=case, seed=seed, all_statistics_exact=True, trajectories_exact=True
            )
        )
    if state_hash(model) != frozen:
        raise AssertionError("Frozen model parameters changed")
    proof = dict(
        parent_commit=PARENT,
        current_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        original_source_lf_sha256=hashlib.sha256(original.encode()).hexdigest(),
        only_validator_changed=True,
        same_process=True,
        checkpoint=fit,
        rows=rows,
    )
    (output / "parent_same_runner.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
