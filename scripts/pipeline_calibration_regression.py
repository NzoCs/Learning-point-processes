"""Verify a repaired calibration against EVERY previously successful draw."""

import argparse
import hashlib
import json
import math
from pathlib import Path

BASELINES = Path("docs/validation/pipeline-repair-20261015")


def draw_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify(baseline, output, reference=None):
    before = json.loads(Path(baseline).read_text(encoding="utf8"))
    old = before["report"]
    references = (
        json.loads(Path(reference).read_text(encoding="utf8"))
        if reference is not None
        else None
    )
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
    previous = equivalent = repaired = total = 0
    max_abs_difference = 0.0
    if [row["hypothesis"] for row in after["results"]] != after["cases"]:
        raise AssertionError("Every declared hypothesis must be present exactly once")
    for row in after["results"]:
        case = row["hypothesis"]
        if row["failed_trials"] or row["completed_trials"] != row["planned_trials"]:
            raise AssertionError(f"Incomplete trials for {case}")
        draws = json.loads((output / f"{case}_draws.json").read_text(encoding="utf8"))
        actual = {str(v["seed"]): draw_hash(v) for v in draws}
        actual_records = {str(v["seed"]): v for v in draws}
        if len(actual) != len(draws) or len(draws) != row["planned_trials"]:
            raise AssertionError("Draws must contain every planned seed exactly once")
        for seed, expected_hash in before["successful_draw_sha256"][case].items():
            if actual.get(seed) != expected_hash:
                if references is None:
                    raise AssertionError(
                        f"Historical observed/null statistics or RNG changed: {case}, seed {seed}"
                    )
                expected = next(v for v in references[case] if str(v["seed"]) == seed)
                if draw_hash(expected) != expected_hash:
                    raise AssertionError(
                        "Numeric reference does not match the original immutable ledger"
                    )
                current = actual_records[seed]
                metadata = lambda v: {
                    k: x for k, x in v.items() if k not in ("observed", "null")
                }
                if metadata(expected) != metadata(current) or len(
                    expected["null"]
                ) != len(current["null"]):
                    raise AssertionError(
                        "Historical RNG, seed or permutation count changed"
                    )
                old_values = [expected["observed"], *expected["null"]]
                new_values = [current["observed"], *current["null"]]
                # The existing public dense-vs-blocked reference checks use
                # exactly these limits; do not tune a threshold to this run.
                if not all(
                    math.isfinite(a)
                    and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
                    for a, b in zip(old_values, new_values)
                ):
                    raise AssertionError(
                        "Historical statistics exceed the existing dense-reference precision contract"
                    )
                max_abs_difference = max(
                    max_abs_difference,
                    max(abs(a - b) for a, b in zip(old_values, new_values)),
                )
                equivalent += 1
            else:
                previous += 1
            if references is not None:
                old_row = next(v for v in old["results"] if v["hypothesis"] == case)
                old_index = next(
                    i for i, v in enumerate(references[case]) if str(v["seed"]) == seed
                )
                new_index = next(
                    i for i, v in enumerate(draws) if str(v["seed"]) == seed
                )
                if old_row["p_values"][old_index] != row["p_values"][new_index]:
                    raise AssertionError("A historical p-value changed")
        for failure in old["errors"]:
            if failure["case"] == case:
                if str(failure["seed"]) not in actual:
                    raise AssertionError(
                        "An original failing seed was silently dropped"
                    )
                repaired += 1
        total += len(draws)
    if previous + equivalent + repaired != total or repaired != len(old["errors"]):
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
        previous_successful_draws_numerically_equivalent=equivalent,
        historical_p_values_exact=(
            previous + equivalent if references is not None else None
        ),
        cross_run_precision=(
            dict(rtol=1e-10, atol=1e-12, max_abs_difference=max_abs_difference)
            if references is not None
            else None
        ),
        original_failures_repaired=repaired,
        total_trials=total,
        comparison="SHA256 first; optional cross-run comparison uses the existing public dense-reference tolerance, with exact RNG metadata and p-values",
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
    parser.add_argument("--numeric-reference", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                BASELINES / f"{args.model}-{args.scenario}_before.json",
                args.output,
                (
                    BASELINES / f"{args.model}-{args.scenario}_reference_draws.json"
                    if args.numeric_reference
                    else None
                ),
            ),
            indent=2,
        )
    )
