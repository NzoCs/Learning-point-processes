"""Ensure a numerical optimization reuses identical pilot data and MMD outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify(original, optimized):
    checked = 0
    for index in range(9):
        a = json.loads((original / f"method-{index}.json").read_text())
        b = json.loads((optimized / f"method-{index}.json").read_text())
        if a["status"] != "cost_probe_complete_not_power" or b["status"] != a["status"]:
            raise ValueError("MMD control incomplete")
        for key in ["observed_sha256", "simulated_sha256"]:
            if a[key] != b[key]:
                raise ValueError("Pilot input data changed")
        left, right = a["tests"][0], b["tests"][0]
        if (
            left["config"] != right["config"]
            or abs(left["p_value"] - right["p_value"]) > 1e-14
        ):
            raise ValueError("Pilot configuration or p-value changed")
        np.testing.assert_allclose(
            left["permuted_statistics"],
            right["permuted_statistics"],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            left["observed_statistic"],
            right["observed_statistic"],
            rtol=1e-10,
            atol=1e-12,
        )
        checked += 1
    return dict(
        status="verified",
        identical_input_pools=True,
        matching_mmd_tests=checked,
        scope="Only deterministic quadrature precomputation and analytic Hawkes compensator changed. Reference mean equivalence is verified separately by tests.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("original", type=Path)
    p.add_argument("optimized", type=Path)
    a = p.parse_args()
    result = verify(a.original, a.optimized)
    with (a.optimized / "replay-verification.json").open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
