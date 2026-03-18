"""
MMD two-sample test on Poisson processes — GPU script.

Three sweeps:
  1. RBF scaling of the SigKernel static kernel
  2. Divergence of the two Poisson processes (varying μ₂)
  3. Batch size

Usage:
    python scripts/mmd_poisson_sweep.py [--device cuda] [--n-repeats 20] [--n-pool 500]
"""

import argparse
import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from typing import List, Dict

from new_ltpp.data.generation.hawkes import HawkesSimulator
from new_ltpp.data.generation.simulation_manager import SimulationManager
from new_ltpp.shared_types import Batch
from new_ltpp.evaluation.statistical_testing import MMDTwoSampleTest
from new_ltpp.evaluation.statistical_testing.point_process_kernels import SIGKernel

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-pool", type=int, default=300, help="Sequences per pool")
    p.add_argument(
        "--n-repeats", type=int, default=10, help="Repetitions per sweep point"
    )
    p.add_argument("--n-permutations", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--batch-size", type=int, default=32, help="Default batch size for sweeps 1 & 2"
    )
    p.add_argument("--end-time", type=float, default=10.0)
    p.add_argument("--mu1", type=float, default=5.0)
    p.add_argument("--mu2", type=float, default=15.0)
    p.add_argument("--out-dir", default="artifacts/mmd_poisson_sweep")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ──────────────────────────────────────────────────────────────────────────────


def simulate_poisson(
    mu: float,
    dim_process: int = 1,
    num_simulations: int = 200,
    end_time: float = 10.0,
    seed=None,
) -> List[Dict]:
    sim = HawkesSimulator(
        mu=[mu] * dim_process,
        alpha=[[0.0] * dim_process] * dim_process,
        beta=[[1.0] * dim_process] * dim_process,
        dim_process=dim_process,
        start_time=0,
        end_time=end_time,
        seed=seed,
    )
    manager = SimulationManager(
        simulation_func=sim.simulate,
        dim_process=dim_process,
        start_time=0,
        end_time=end_time,
    )
    return manager.bulk_simulate(num_simulations)


def sequences_to_batch(
    sequences: List[Dict], max_len: int | None = None, device: str = "cpu"
) -> Batch:
    if max_len is None:
        max_len = max(s["seq_len"] for s in sequences)
    time_seqs, delta_seqs, type_seqs, masks = [], [], [], []
    for s in sequences:
        L = s["seq_len"]
        time_seqs.append(s["time_since_start"] + [0.0] * (max_len - L))
        delta_seqs.append(s["time_since_last_event"] + [0.0] * (max_len - L))
        type_seqs.append(s["type_event"] + [0] * (max_len - L))
        masks.append([True] * L + [False] * (max_len - L))
    return Batch(
        time_seqs=torch.tensor(time_seqs, dtype=torch.float32, device=device),
        time_delta_seqs=torch.tensor(delta_seqs, dtype=torch.float32, device=device),
        type_seqs=torch.tensor(type_seqs, dtype=torch.long, device=device),
        valid_event_mask=torch.tensor(masks, dtype=torch.bool, device=device),
    )


def subsample_batch(
    seqs_pool: List[Dict], n: int, max_len: int, device: str = "cpu"
) -> Batch:
    idx = np.random.choice(len(seqs_pool), n, replace=True)
    return sequences_to_batch([seqs_pool[i] for i in idx], max_len, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────


def run_one_test(test: MMDTwoSampleTest, batch_x: Batch, batch_y: Batch):
    observed, perm_mmds = test._permutation_test_from_batches(batch_x, batch_y)
    p_val = (sum(1 for pm in perm_mmds if pm >= observed) + 1) / (len(perm_mmds) + 1)
    return observed, perm_mmds, p_val


def empirical_analysis(
    kernel,
    seqs_X,
    seqs_Y,
    max_len,
    device,
    n_repeats,
    batch_size,
    n_permutations,
    alpha,
) -> Dict:
    test = MMDTwoSampleTest(kernel=kernel, n_permutations=n_permutations)
    h0_mmds, h0_pvals, h0_perm_dists = [], [], []
    h1_mmds, h1_pvals, h1_perm_dists = [], [], []

    for _ in range(n_repeats):
        bx = subsample_batch(seqs_X, batch_size, max_len, device)
        by = subsample_batch(seqs_X, batch_size, max_len, device)
        obs, perms, pval = run_one_test(test, bx, by)
        h0_mmds.append(obs)
        h0_perm_dists.extend(perms)
        h0_pvals.append(pval)

        ba = subsample_batch(seqs_X, batch_size, max_len, device)
        bb = subsample_batch(seqs_Y, batch_size, max_len, device)
        obs, perms, pval = run_one_test(test, ba, bb)
        h1_mmds.append(obs)
        h1_perm_dists.extend(perms)
        h1_pvals.append(pval)

    return {
        "h0_mmds": h0_mmds,
        "h0_pvals": h0_pvals,
        "h0_perm_dists": h0_perm_dists,
        "h1_mmds": h1_mmds,
        "h1_pvals": h1_pvals,
        "h1_perm_dists": h1_perm_dists,
        "type1_error": float(np.mean([p < alpha for p in h0_pvals])),
        "type2_error": float(np.mean([p >= alpha for p in h1_pvals])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_sweep(sweep_values, results_list, param_name, param_label, alpha, out_dir):
    n = len(sweep_values)

    # Fig 1 : MMD² distributions
    fig1, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]
    for i, (val, res) in enumerate(zip(sweep_values, results_list)):
        ax = axes[i]
        ax.hist(
            res["h0_perm_dists"],
            bins=40,
            color="gray",
            alpha=0.5,
            density=True,
            label="Null (perms H0)",
        )
        thresh = float(np.percentile(res["h0_perm_dists"], 100 * (1 - alpha)))
        ax.axvline(
            thresh,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Seuil α={alpha}",
        )
        for mmd in res["h0_mmds"]:
            ax.axvline(mmd, color="steelblue", alpha=0.35, linewidth=1)
        h0_mean = float(np.mean(res["h0_mmds"]))
        ax.axvline(
            h0_mean, color="steelblue", linewidth=2.5, label=f"H0 moy={h0_mean:.4f}"
        )
        for mmd in res["h1_mmds"]:
            ax.axvline(mmd, color="crimson", alpha=0.35, linewidth=1)
        h1_mean = float(np.mean(res["h1_mmds"]))
        ax.axvline(
            h1_mean, color="crimson", linewidth=2.5, label=f"H1 moy={h1_mean:.4f}"
        )
        ax.set_title(f"{param_label}={val}", fontsize=11)
        ax.set_xlabel("MMD²", fontsize=10)
        if i == 0:
            ax.set_ylabel("Densité", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    fig1.suptitle(
        f"Distributions MMD² — sweep {param_name}", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path1 = os.path.join(out_dir, f"sweep_{param_name}_distributions.png")
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # Fig 2 : error curves
    type1_errors = [r["type1_error"] for r in results_list]
    type2_errors = [r["type2_error"] for r in results_list]
    xvals = list(range(n))
    xlabels = [str(v) for v in sweep_values]
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(xvals, type1_errors, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.axhline(y=alpha, color="red", linestyle="--", alpha=0.8, label=f"α={alpha}")
    ax1.set_xticks(xvals)
    ax1.set_xticklabels(xlabels, rotation=20)
    ax1.set_xlabel(param_label, fontsize=11)
    ax1.set_ylabel("Erreur Type 1", fontsize=11)
    ax1.set_title(f"Erreur Type 1 vs {param_name}", fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(xvals, type2_errors, "s-", color="crimson", linewidth=2, markersize=8)
    ax2.set_xticks(xvals)
    ax2.set_xticklabels(xlabels, rotation=20)
    ax2.set_xlabel(param_label, fontsize=11)
    ax2.set_ylabel("Erreur Type 2", fontsize=11)
    ax2.set_title(f"Erreur Type 2 vs {param_name}", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    fig2.suptitle(
        f"Erreurs empiriques — sweep {param_name}", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path2 = os.path.join(out_dir, f"sweep_{param_name}_errors.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {path2}")


def make_kernel(rbf_scaling: float, dim: int) -> SIGKernel:
    return SIGKernel(
        static_kernel_type="rbf",
        embedding_type="linear_interpolant",
        dyadic_order=3,
        num_event_types=dim,
        rbf_scaling=rbf_scaling,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DIM = 1

    print(f"Device : {args.device}")
    print(f"μ₁={args.mu1}  μ₂={args.mu2}  end_time={args.end_time}")
    print(
        f"N_pool={args.n_pool}  N_repeats={args.n_repeats}  N_perms={args.n_permutations}"
    )
    print(f"Output : {args.out_dir}\n")

    # ── Generate base pools ────────────────────────────────────────────────
    print("Generating pools...")
    seqs_A = simulate_poisson(
        args.mu1,
        dim_process=DIM,
        num_simulations=args.n_pool,
        end_time=args.end_time,
        seed=42,
    )
    seqs_B = simulate_poisson(
        args.mu2,
        dim_process=DIM,
        num_simulations=args.n_pool,
        end_time=args.end_time,
        seed=123,
    )
    global_max_len = max(s["seq_len"] for s in seqs_A + seqs_B)
    print(
        f"  Pool A : {len(seqs_A)} seqs, mean_len={np.mean([s['seq_len'] for s in seqs_A]):.1f}"
    )
    print(
        f"  Pool B : {len(seqs_B)} seqs, mean_len={np.mean([s['seq_len'] for s in seqs_B]):.1f}"
    )
    print(f"  global_max_len={global_max_len}\n")

    kw = dict(
        device=args.device,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
    )

    # ── Sweep 1 : rbf_scaling ─────────────────────────────────────────────
    RBF_SCALINGS = [0.1, 0.3, 1.0, 3.0, 10.0]
    print("=== Sweep 1 : rbf_scaling ===")
    rbf_results = []
    for s in RBF_SCALINGS:
        print(f"  rbf_scaling={s} ...", end=" ", flush=True)
        res = empirical_analysis(
            make_kernel(s, DIM),
            seqs_A,
            seqs_B,
            global_max_len,
            **kw,
        )
        rbf_results.append(res)
        print(f"Type1={res['type1_error']:.2f}  Type2={res['type2_error']:.2f}")
    plot_sweep(
        RBF_SCALINGS,
        rbf_results,
        "rbf_scaling",
        "rbf_scaling",
        args.alpha,
        args.out_dir,
    )

    # ── Sweep 2 : μ₂ divergence ───────────────────────────────────────────
    MU_2_VALUES = [args.mu1, 7.0, 10.0, 15.0, 20.0, 30.0]
    print("\n=== Sweep 2 : μ₂ ===")
    kernel_fixed = make_kernel(1.0, DIM)
    mu_results = []
    for mu2 in MU_2_VALUES:
        print(f"  μ₂={mu2} ...", end=" ", flush=True)
        seqs_B2 = simulate_poisson(
            mu2, dim_process=DIM, num_simulations=args.n_pool, end_time=args.end_time
        )
        max_len2 = max(s["seq_len"] for s in seqs_A + seqs_B2)
        res = empirical_analysis(kernel_fixed, seqs_A, seqs_B2, max_len2, **kw)
        mu_results.append(res)
        print(f"Type1={res['type1_error']:.2f}  Type2={res['type2_error']:.2f}")
    plot_sweep(MU_2_VALUES, mu_results, "mu2", "μ₂", args.alpha, args.out_dir)

    # ── Sweep 3 : batch size ──────────────────────────────────────────────
    BATCH_SIZES = [8, 16, 32, 64]
    print("\n=== Sweep 3 : batch_size ===")
    bs_results = []
    for bs in BATCH_SIZES:
        print(f"  batch_size={bs} ...", end=" ", flush=True)
        res = empirical_analysis(
            kernel_fixed,
            seqs_A,
            seqs_B,
            global_max_len,
            device=args.device,
            n_repeats=args.n_repeats,
            batch_size=bs,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
        )
        bs_results.append(res)
        print(f"Type1={res['type1_error']:.2f}  Type2={res['type2_error']:.2f}")
    plot_sweep(
        BATCH_SIZES,
        bs_results,
        "batch_size",
        "Taille de batch",
        args.alpha,
        args.out_dir,
    )

    print(f"\nAll done. Figures saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
