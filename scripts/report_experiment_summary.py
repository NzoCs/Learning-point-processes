"""Strict aggregation of all 20 prespecified scientific experiment cells."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta, binomtest

from scripts.report_experiment_validation import (
    CASES,
    FAMILY_SIZE,
    METHODS,
    read_records,
    seed_entropy,
    summarize,
    write_json,
)
from new_ltpp.evaluation.statistical_testing.statistical_tests.mmd_test import (
    monte_carlo_p_value,
)


def holm_adjust(ps, family_size=None):
    family_size = len(ps) if family_size is None else family_size
    if family_size < len(ps):
        raise ValueError("Family cannot be smaller than the observed comparisons")
    order = np.argsort(ps)
    adjusted = np.empty(len(ps))
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (family_size - rank) * ps[index])
        adjusted[index] = min(1.0, maximum)
    return adjusted.tolist()


def aggregate(root):
    root = Path(root)
    comparisons, calibration, errors = [], [], []
    provenance = None
    for case in CASES:
        combined, pilot = [], None
        for cohort in (0, 1):
            directory = root / case / f"cohort-{cohort}"
            try:
                report = json.loads((directory / "report.json").read_text())
                plan = report["design"]
                count = 1000 if case.endswith("H0") else 250
                expected = dict(
                    n=128,
                    trials=count,
                    permutations=499,
                    grid=100,
                    horizon=1.0,
                    burn=10.0,
                    case=case,
                    cohort=cohort,
                    evaluation_domain=0,
                )
                if any(plan.get(k) != v for k, v in expected.items()):
                    raise ValueError("Cell is not the prespecified primary design")
                if report["status"] != "completed" or report["errors"]:
                    raise ValueError("Incomplete/failed cell")
                if not plan["source_committed"] or plan["tracked_dirty"]:
                    raise ValueError(
                        "Primary campaign requires committed, clean source"
                    )
                if pilot is not None and pilot != plan["pilot"]:
                    raise ValueError(
                        "Cohorts did not use the same frozen independent pilot"
                    )
                pilot = plan["pilot"]
                signature = {
                    k: plan[k]
                    for k in (
                        "commit",
                        "source_lf_sha256",
                        "lock_sha256",
                        "numpy",
                        "torch",
                        "pysiglib",
                        "device",
                        "methods",
                        "master_seed",
                    )
                }
                if provenance is not None and signature != provenance:
                    raise ValueError("Mixed source/environment/methods")
                provenance = signature
                records = read_records(directory / "trials.jsonl")
                if len(records) != count or any(
                    r["status"] != "completed" for r in records
                ):
                    raise ValueError("Missing/failed scientific trials")
                for row in records:
                    expected_seed = int(
                        np.random.SeedSequence(
                            seed_entropy(case, cohort, row["trial"], 3)
                        ).generate_state(1, dtype=np.uint64)[0]
                    ) % (2**63)
                    if row["swap_seed"] != expected_seed:
                        raise ValueError(
                            "Raw draws belong to another case/cohort/domain"
                        )
                    for method in METHODS:
                        stat = row["methods"][method]
                        if len(stat["null"]) != 499:
                            raise ValueError("Incomplete randomization")
                        p = float(
                            monte_carlo_p_value(
                                torch.tensor(stat["observed"], dtype=torch.float64),
                                torch.tensor(stat["null"], dtype=torch.float64),
                            )
                        )
                        if p != stat["p_value"]:
                            raise ValueError(
                                "Stored p-value disagrees with raw statistics"
                            )
                        if plan["same_counts"] and method == "count_rbf" and p != 1.0:
                            raise ValueError("Count-matching control was broken")
                actual = summarize(case, records, count)
                if actual != report["results"]:
                    raise ValueError("Reported summary disagrees with raw draws")
                if case.endswith("H0"):
                    calibration.extend(
                        dict(case=case, cohort=cohort, **r) for r in actual
                    )
                combined.extend(records)
            except Exception as error:
                errors.append(dict(case=case, cohort=cohort, error=repr(error)))
        if len(combined) != (2000 if case.endswith("H0") else 500):
            continue
        pooled = summarize(case, combined, len(combined))
        if case.endswith("H0"):
            for r in pooled:
                k, n = r["rejected"], r["valid_trials"]
                # Simultaneous one-sided bounds over the 12 pooled H0 cells.
                r["simultaneous_upper_99"] = (
                    float(beta.ppf(1 - 0.01 / 12, k + 1, n - k)) if k < n else 1.0
                )
                r["excludes_inflation_above_7_5_percent"] = (
                    r["simultaneous_upper_99"] < 0.075
                )
                calibration.append(dict(case=case, cohort="pooled", **r))
        else:
            for a, b in itertools.combinations(METHODS, 2):
                a_only = sum(
                    r["methods"][a]["p_value"] <= 0.05
                    and r["methods"][b]["p_value"] > 0.05
                    for r in combined
                )
                b_only = sum(
                    r["methods"][b]["p_value"] <= 0.05
                    and r["methods"][a]["p_value"] > 0.05
                    for r in combined
                )
                p = (
                    float(binomtest(a_only, a_only + b_only, 0.5).pvalue)
                    if a_only + b_only
                    else 1.0
                )
                comparisons.append(
                    dict(
                        case=case,
                        method_a=a,
                        method_b=b,
                        a_only=a_only,
                        b_only=b_only,
                        paired_exact_p=p,
                    )
                )
        write_json(root / f"pooled-{case}.json", pooled)
    adjusted = holm_adjust([r["paired_exact_p"] for r in comparisons], family_size=18)
    for row, p in zip(comparisons, adjusted):
        row["holm_p_over_18_comparisons"] = p
    excess = any(r["h0_excess_detected"] for r in calibration)
    pooled_h0 = [r for r in calibration if r["cohort"] == "pooled"]
    complete = not errors and len(calibration) == FAMILY_SIZE and len(comparisons) == 18
    result = dict(
        status="complete" if complete else "incomplete",
        provenance=provenance,
        errors=errors,
        calibration=calibration,
        paired_power_comparisons=comparisons,
        h0_excess_detected=excess,
        precision_sufficient_for_all_pooled_cells=(
            complete
            and all(r["excludes_inflation_above_7_5_percent"] for r in pooled_h0)
        ),
        limitations=[
            "No excess detected is not a proof that size equals exactly 5%.",
            "A precision limit of 7.5% is prespecified; an inconclusive bound is not itself a model/code defect.",
            "Fixed-window, unmarked synthetic processes; learned models, quotas and termination use the separate production-pipeline campaign.",
            "Power-law processes start from empty history at -10; no exact stationarity claim.",
        ],
    )
    write_json(root / "summary.json", result)
    if not complete or excess:
        raise RuntimeError("Campaign incomplete or H0 excess; see summary.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    aggregate(parser.parse_args().root)
