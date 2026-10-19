"""Independent, training-only maximum likelihood for frozen RMSE references.

The family is prespecified; no generative parameter or evaluation sample is an
argument to the fitter. Likelihood conditions on empty visible history at zero.
"""

from dataclasses import asdict

import numpy as np
from scipy.optimize import minimize

from scripts.report_experiment_validation import Process, event_hash
from scripts.report_rmse_validation import check_rows

STARTS = ((0.2, 1.0, 1.5), (0.5, 10.0, 2.0), (0.8, 100.0, 4.0))
BOUNDS = ((-18.0, 12.0), (0.0, 0.98), (-5.0, 7.0), (-3.0, 2.2))


def likelihood_data(rows):
    check_rows(rows, 1.0)
    ages, destinations, remaining = [], [], []
    offset = 0
    for row in rows:
        if np.any(np.diff(row) <= 0):
            raise ValueError(
                "Strictly increasing simple-process training times required"
            )
        i, j = np.tril_indices(len(row), -1)
        ages.append(row[i] - row[j])
        destinations.append(i + offset)
        remaining.append(1 - row)
        offset += len(row)
    return (
        np.concatenate(ages),
        np.concatenate(destinations),
        np.concatenate(remaining),
        len(rows),
    )


def nll_gradient(theta, data, kind):
    """Exact integrated Hawkes likelihood and analytic gradient per trajectory."""
    ages, index, remaining, n = data
    mu, eta, beta = np.exp(theta[0]), theta[1], np.exp(theta[2])
    if kind == "exp":
        k = beta * np.exp(-beta * ages)
        dk_beta = k * (1 - beta * ages)  # derivative wrt log(beta)
        tail = np.exp(-beta * remaining)
        mass = -np.expm1(-beta * remaining)
        dm_beta = beta * remaining * tail
    elif kind == "power":
        p_minus_one = np.exp(theta[3])
        exponent = 1 + p_minus_one
        base = 1 + beta * ages
        log_base = np.log1p(beta * ages)
        k = beta * p_minus_one * np.exp(-exponent * log_base)
        dk_beta = k * (1 - exponent * beta * ages / base)
        dk_p = k * (1 - p_minus_one * log_base)
        log_tail = np.log1p(beta * remaining)
        tail = np.exp(-p_minus_one * log_tail)
        mass = -np.expm1(-p_minus_one * log_tail)
        dm_beta = p_minus_one * beta * remaining / (1 + beta * remaining) * tail
        dm_p = p_minus_one * log_tail * tail
    else:
        raise ValueError("Hawkes family must be exp or power")
    summed = np.bincount(index, weights=k, minlength=len(remaining))
    intensity = mu + eta * summed
    value = n * mu + eta * mass.sum() - np.log(intensity).sum()
    grad = [
        mu * (n - (1 / intensity).sum()),
        mass.sum() - (summed / intensity).sum(),
        eta * (dm_beta.sum() - (dk_beta / intensity[index]).sum()),
    ]
    if kind == "power":
        grad.append(eta * (dm_p.sum() - (dk_p / intensity[index]).sum()))
    return float(value / n), np.asarray(grad) / n


def fit_reference(rows, kind):
    """Fit all free parameters from training observations only, without RNG."""
    if kind not in ("poisson", "exp", "power"):
        raise ValueError("Unknown training family")
    data = likelihood_data(rows)
    empirical_rate = len(data[2]) / len(rows)
    if empirical_rate == 0:
        raise ValueError("No training events: no positive finite-rate MLE")
    if kind == "poisson":
        return dict(
            reference=asdict(Process("poisson", rate=empirical_rate)),
            training_sha256=event_hash(rows),
            training_paths=len(rows),
            training_events=len(data[2]),
            objective="count / exposure (exact Poisson MLE)",
            optimizer_runs=[],
            boundary_parameters=[],
        )
    runs = []
    dimension = 4 if kind == "power" else 3
    for eta, beta, exponent in STARTS:
        initial = [np.log(empirical_rate * (1 - eta)), eta, np.log(beta)]
        if kind == "power":
            initial.append(np.log(exponent - 1))
        result = minimize(
            nll_gradient,
            initial,
            args=(data, kind),
            jac=True,
            method="L-BFGS-B",
            bounds=BOUNDS[:dimension],
            options=dict(maxiter=500, ftol=1e-12, gtol=1e-6, maxls=40),
        )
        runs.append(
            dict(
                success=bool(result.success),
                message=str(result.message),
                objective=float(result.fun),
                theta=result.x.tolist(),
                iterations=int(result.nit),
            )
        )
    converged = [r for r in runs if r["success"] and np.isfinite(r["objective"])]
    if not converged:
        raise RuntimeError(f"No converged training fit: {runs}")
    best = min(converged, key=lambda r: r["objective"])
    theta = np.asarray(best["theta"])
    mu, eta, beta = np.exp(theta[0]), theta[1], np.exp(theta[2])
    reference = Process(
        kind,
        rate=mu / (1 - eta),
        branching=eta,
        beta=beta,
        exponent=1 + np.exp(theta[3]) if kind == "power" else 2.0,
    )
    return dict(
        reference=asdict(reference),
        training_sha256=event_hash(rows),
        training_paths=len(rows),
        training_events=len(data[2]),
        objective="visible-history conditional negative log likelihood per trajectory",
        optimizer_runs=runs,
        selected_start=runs.index(best),
        boundary_parameters=[
            i
            for i, (v, bounds) in enumerate(zip(theta, BOUNDS))
            if min(abs(v - bounds[0]), abs(v - bounds[1])) < 1e-5
        ],
    )
