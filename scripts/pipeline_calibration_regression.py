"""Verify a repaired calibration against EVERY previously successful draw."""

import argparse
import hashlib
import json
from pathlib import Path

BASELINES = Path("docs/validation/pipeline-repair-20261015")


def draw_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify(baseline, output):
    before = json.loads(Path(baseline).read_text(encoding="utf8"))
    old = before["report"]
    output = Path(output)
    after = json.loads((output / "report.json").read_text(encoding="utf8"))
    if after["status"] != "completed" or after["errors"]:
        raise AssertionError(
            "Repaired campaign must complete without discarded failures"
        )
    if not after["model_frozen_verified"] or not after["no_h0_excess_detected"]:
        raise AssertionError("Frozen model and declared H0 criterion must pass")
    for key in (
        "model",
        "scenario",
        "seed",
        "event_quota",
        "batch_sizes",
        "history_events",
        "marks",
        "H0_trials",
        "H1_trials",
        "cases",
        "kernel_config",
        "protocol",
        "observation",
        "h0_family_size",
        "h0_excess_family_threshold",
    ):
        if old[key] != after[key]:
            raise AssertionError(f"Scientific protocol changed: {key}")
    if old["fit"] != after["fit"]:
        raise AssertionError(
            "Frozen checkpoint, parameters, training metadata or sampler settings changed"
        )
    previous = repaired = total = 0
    if [row["hypothesis"] for row in after["results"]] != after["cases"]:
        raise AssertionError("Every declared hypothesis must be present exactly once")
    for row in after["results"]:
        case = row["hypothesis"]
        if row["failed_trials"] or row["completed_trials"] != row["planned_trials"]:
            raise AssertionError(f"Incomplete trials for {case}")
        draws = json.loads((output / f"{case}_draws.json").read_text(encoding="utf8"))
        actual = {str(v["seed"]): draw_hash(v) for v in draws}
        if len(actual) != len(draws) or len(draws) != row["planned_trials"]:
            raise AssertionError("Draws must contain every planned seed exactly once")
        for seed, expected_hash in before["successful_draw_sha256"][case].items():
            if actual.get(seed) != expected_hash:
                raise AssertionError(
                    f"Historical observed/null statistics or RNG changed: {case}, seed {seed}"
                )
            previous += 1
        for failure in old["errors"]:
            if failure["case"] == case:
                if str(failure["seed"]) not in actual:
                    raise AssertionError(
                        "An original failing seed was silently dropped"
                    )
                repaired += 1
        total += len(draws)
    if previous + repaired != total or repaired != len(old["errors"]):
        raise AssertionError(
            "Successful and previously failing trials must partition the rerun"
        )
    result = dict(
        model=after["model"],
        scenario=after["scenario"],
        completed=True,
        baseline_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
        source_commit=after["code"]["commit"],
        checkpoint_unchanged=True,
        scientific_protocol_unchanged=True,
        previous_successful_draws_exact=previous,
        original_failures_repaired=repaired,
        total_trials=total,
        comparison="Canonical SHA256 of seed, observed statistic, all 99 null statistics and null RNG seed; no numerical tolerance",
    )
    (output / "historical_regression.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                BASELINES / f"{args.model}-{args.scenario}_before.json", args.output
            ),
            indent=2,
        )
    )
