"""
Test MMD sur des processus de Hawkes avec différents paramètres

Ce script génère des processus de Hawkes avec différentes configurations
et les compare via le test MMD two-sample avec MKernel (RBF+Exponential) et SigKernel.

Usage:
    python test_mmd_hawkes.py [--num-sim 200] [--n-permutations 200] [--batch-size 64]
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations
from datetime import datetime

import numpy as np
import torch
import pandas as pd

from new_ltpp.data.generation.hawkes import HawkesSimulator
from new_ltpp.data.generation.simulation_manager import SimulationManager
from new_ltpp.shared_types import Batch
from new_ltpp.evaluation.statistical_testing import MMDTwoSampleTest
from new_ltpp.evaluation.statistical_testing.kernels import (
    MKernel,
    MKernelTransform,
    create_time_kernel,
    EmbeddingKernel,
    # SIGKernel,
)


# ============================================================================
# Configuration des paramètres Hawkes
# ============================================================================


def get_hawkes_configs(dim_process: int = 2, end_time: int = 30) -> Dict[str, dict]:
    """Définit les configurations de processus de Hawkes à tester."""
    return {
        "baseline": dict(
            mu=[0.2, 0.2],
            alpha=[[0.3, 0.1], [0.1, 0.3]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "baseline_copy": dict(
            mu=[0.2, 0.2],
            alpha=[[0.3, 0.1], [0.1, 0.3]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "high_excitation": dict(
            mu=[0.2, 0.2],
            alpha=[[0.8, 0.5], [0.5, 0.8]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "low_excitation": dict(
            mu=[0.2, 0.2],
            alpha=[[0.05, 0.02], [0.02, 0.05]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "high_baseline": dict(
            mu=[0.8, 0.8],
            alpha=[[0.3, 0.1], [0.1, 0.3]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "slow_decay": dict(
            mu=[0.2, 0.2],
            alpha=[[0.3, 0.1], [0.1, 0.3]],
            beta=[[0.5, 0.3], [0.3, 0.5]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
        "asymmetric": dict(
            mu=[0.1, 0.5],
            alpha=[[0.1, 0.6], [0.05, 0.2]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        ),
    }


# ============================================================================
# Helpers pour simulation et conversion
# ============================================================================


def simulate_hawkes(
    mu,
    alpha,
    beta,
    dim_process,
    num_simulations=200,
    start_time=0,
    end_time=30,
    seed=None,
) -> List[Dict]:
    """Simule des processus de Hawkes."""
    sim = HawkesSimulator(
        mu=mu,
        alpha=alpha,
        beta=beta,
        dim_process=dim_process,
        start_time=start_time,
        end_time=end_time,
        seed=seed,
    )
    manager = SimulationManager(
        simulation_func=sim.simulate,
        dim_process=dim_process,
        start_time=start_time,
        end_time=end_time,
    )
    return manager.bulk_simulate(num_simulations)


def sequences_to_batch(sequences: List[Dict], max_len: int = None) -> Batch:
    """Convertit une liste de séquences en Batch (padded)."""
    if max_len is None:
        max_len = max(s["seq_len"] for s in sequences)

    time_seqs, delta_seqs, type_seqs, masks = [], [], [], []

    for s in sequences:
        L = s["seq_len"]
        t = s["time_since_start"] + [0.0] * (max_len - L)
        dt = s["time_since_last_event"] + [0.0] * (max_len - L)
        tp = s["type_event"] + [0] * (max_len - L)
        m = [True] * L + [False] * (max_len - L)

        time_seqs.append(t)
        delta_seqs.append(dt)
        type_seqs.append(tp)
        masks.append(m)

    return Batch(
        time_seqs=torch.tensor(time_seqs, dtype=torch.float32),
        time_delta_seqs=torch.tensor(delta_seqs, dtype=torch.float32),
        type_seqs=torch.tensor(type_seqs, dtype=torch.long),
        valid_event_mask=torch.tensor(masks, dtype=torch.bool),
    )


def split_batch(batch: Batch, batch_size: int) -> List[Batch]:
    """Découpe un gros Batch en mini-batches."""
    n = batch.time_seqs.shape[0]
    sub_batches = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        sub_batches.append(
            Batch(
                time_seqs=batch.time_seqs[i:end],
                time_delta_seqs=batch.time_delta_seqs[i:end],
                type_seqs=batch.type_seqs[i:end],
                valid_event_mask=batch.valid_event_mask[i:end],
            )
        )
    return sub_batches


def estimate_memory_gb(batch_size: int, seq_len: int) -> float:
    """Estimate peak memory usage for kernel computation in GB.
    
    The kernel creates 4D tensors of shape (B, B, L, L) for intra-batch
    and (B1, B2, L, K) for cross-batch computations.
    Multiple such tensors exist simultaneously.
    """
    # 3 kernel matrix calls in MMD (k_xx, k_yy, k_xy)
    # Each creates ~3 intermediate 4D tensors
    elements_per_tensor = batch_size * batch_size * seq_len * seq_len
    bytes_per_tensor = elements_per_tensor * 4  # float32
    # ~3 tensors alive at peak during one gram matrix computation
    peak_bytes = bytes_per_tensor * 3
    return peak_bytes / (1024 ** 3)


def mmd_test_by_batch(
    batch_x: Batch,
    batch_y: Batch,
    test: MMDTwoSampleTest,
    batch_size: int = 16,
    verbose: bool = False,
) -> Dict:
    """Exécute le test MMD par batch et agrège les résultats."""
    sub_x = split_batch(batch_x, batch_size)
    sub_y = split_batch(batch_y, batch_size)
    n_pairs = min(len(sub_x), len(sub_y))

    mmd_values = []
    p_values = []

    for i in range(n_pairs):
        if verbose:
            print(f"  Batch {i + 1}/{n_pairs}...", end=" ", flush=True)
        mmd_val = test.statistic_from_batches(sub_x[i], sub_y[i])
        p_val = test.p_value_from_batches(sub_x[i], sub_y[i])
        mmd_values.append(mmd_val)
        p_values.append(p_val)
        if verbose:
            print(
                f"MMD² = {mmd_val:.6f}, p-value = {p_val:.4f}"
            )

    return {
        "mmd_values": mmd_values,
        "p_values": p_values,
        "mean_mmd": np.mean(mmd_values),
        "mean_p_value": np.mean(p_values),
    }


# ============================================================================
# Génération des données
# ============================================================================


def generate_all_configs(configs: Dict[str, dict], num_sim: int) -> Tuple[Dict, int]:
    """Génère toutes les configurations et retourne les batches."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES DONNÉES")
    print("=" * 80)

    raw_sequences = {}
    for name, params in configs.items():
        print(f"Simulation de '{name}'...")
        raw_sequences[name] = simulate_hawkes(**params, num_simulations=num_sim)
        avg_len = np.mean([s["seq_len"] for s in raw_sequences[name]])
        print(
            f"  → {len(raw_sequences[name])} séquences, longueur moyenne = {avg_len:.1f}"
        )

    # Padding commun
    global_max_len = max(s["seq_len"] for seqs in raw_sequences.values() for s in seqs)
    print(f"\nMax seq length (padding commun) : {global_max_len}")

    batches = {}
    for name, seqs in raw_sequences.items():
        batches[name] = sequences_to_batch(seqs, max_len=global_max_len)
        print(f"Batch '{name}' : {batches[name].time_seqs.shape}")

    return batches, global_max_len


# ============================================================================
# Configuration des kernels
# ============================================================================


def setup_kernels(dim_process: int, n_permutations: int) -> Tuple:
    """Configure les kernels MMD."""
    print("\n" + "=" * 80)
    print("CONFIGURATION DES KERNELS")
    print("=" * 80)

    # MKernel (RBF time kernel + Exponential transform)
    # Reduced sigma for more sensitivity to differences
    time_kernel_rbf = create_time_kernel("rbf")
    type_kernel = EmbeddingKernel(num_classes=dim_process, embedding_dim=8, sigma=1.0)

    mkernel = MKernel(
        time_kernel=time_kernel_rbf,
        type_kernel=type_kernel,
        transform=MKernelTransform.EXPONENTIAL,
    )

    # SigKernel
    # sigkernel = SIGKernel(
    #     static_kernel_type="rbf",
    #     embedding_type="linear_interpolant",
    #     dyadic_order=3,
    #     num_event_types=dim_process,
    #     sigma=1.0,
    # )

    # Tests MMD
    mmd_test_mk = MMDTwoSampleTest(kernel=mkernel, n_permutations=n_permutations)
    # mmd_test_sig = MMDTwoSampleTest(kernel=sigkernel, n_permutations=n_permutations)

    print(
        f"  1. MKernel : RBF time + Embedding type, transform={mkernel.transform.value}"
    )
    # print(
    #     f"  2. SIGKernel : static_kernel={sigkernel.static_kernel_type}, dyadic_order={sigkernel.dyadic_order}"
    # )
    print(f"  Permutations : {n_permutations}")

    return mmd_test_mk, None  # mmd_test_sig


# ============================================================================
# Test sanity check (same vs same)
# ============================================================================


def run_sanity_check(batches: Dict, mmd_test_mk, mmd_test_sig, batch_size: int):
    """Exécute le sanity check (same vs same)."""
    print("\n" + "=" * 80)
    print("SANITY CHECK : baseline vs baseline_copy")
    print("=" * 80)

    print("\n📊 Test avec MKernel (RBF + Exponential):")
    print("-" * 80)
    result_mk = mmd_test_by_batch(
        batches["baseline"],
        batches["baseline_copy"],
        mmd_test_mk,
        batch_size,
        verbose=True,
    )
    print(f"\n→ MMD² moyen = {result_mk['mean_mmd']:.6f}")
    print(f"→ p-value moyenne = {result_mk['mean_p_value']:.4f}")

    # print("\n📊 Test avec SigKernel:")
    # print("-" * 80)
    # result_sig = mmd_test_by_batch(
    #     batches["baseline"],
    #     batches["baseline_copy"],
    #     mmd_test_sig,
    #     batch_size,
    #     verbose=True,
    # )
    # print(f"\n→ MMD² moyen = {result_sig['mean_mmd']:.6f}")
    # print(f"→ p-value moyenne = {result_sig['mean_p_value']:.4f}")

    return result_mk, None  # result_sig


# ============================================================================
# Tests entre toutes les paires
# ============================================================================


def run_pairwise_tests(
    configs: Dict, batches: Dict, mmd_test_mk, mmd_test_sig, batch_size: int
):
    """Teste toutes les paires de configurations."""
    print("\n" + "=" * 80)
    print("TESTS ENTRE PAIRES DE DISTRIBUTIONS")
    print("=" * 80)

    config_names = [n for n in configs.keys() if n != "baseline_copy"]
    pairs = list(combinations(config_names, 2))

    results_mk = {}
    results_sig = {}

    for name_a, name_b in pairs:
        print(f"\n{'─' * 60}")
        print(f"Test : {name_a} vs {name_b}")
        print(f"{'─' * 60}")

        # MKernel
        print("  🔹 MKernel (RBF + Exponential)...")
        res_mk = mmd_test_by_batch(
            batches[name_a], batches[name_b], mmd_test_mk, batch_size, verbose=False
        )
        results_mk[(name_a, name_b)] = res_mk
        rejected_mk = res_mk["mean_p_value"] < 0.05
        print(
            f"    → MMD² = {res_mk['mean_mmd']:.6f}, p-value = {res_mk['mean_p_value']:.4f}"
        )
        print(f"    → {'H0 REJETÉE' if rejected_mk else 'H0 non rejetée'}")

        # SigKernel
        # print("  🔹 SigKernel...")
        # res_sig = mmd_test_by_batch(
        #     batches[name_a], batches[name_b], mmd_test_sig, batch_size, verbose=False
        # )
        # results_sig[(name_a, name_b)] = res_sig
        # rejected_sig = res_sig["mean_p_value"] < 0.05
        # print(
        #     f"    → MMD² = {res_sig['mean_mmd']:.6f}, p-value = {res_sig['mean_p_value']:.4f}"
        # )
        # print(f"    → {'H0 REJETÉE' if rejected_sig else 'H0 non rejetée'}")

    return results_mk, {}  # results_sig


# ============================================================================
# Étude paramétrique sur alpha
# ============================================================================


def study_alpha_variation(
    baseline_batch: Batch,
    mmd_test_mk,
    mmd_test_sig,
    dim_process: int,
    num_sim: int,
    max_len: int,
    end_time: int = 30,
):
    """Étudie la variation du paramètre alpha."""
    print("\n" + "=" * 80)
    print("ÉTUDE PARAMÉTRIQUE : VARIATION DE ALPHA")
    print("=" * 80)

    alpha_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    alpha_results_mk = []
    alpha_results_sig = []

    for alpha_diag in alpha_values:
        off_diag = alpha_diag * 0.3
        params = dict(
            mu=[0.2, 0.2],
            alpha=[[alpha_diag, off_diag], [off_diag, alpha_diag]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        )
        seqs = simulate_hawkes(**params, num_simulations=num_sim)
        batch_alpha = sequences_to_batch(seqs, max_len=max_len)

        # MKernel
        mmd_val_mk = mmd_test_mk.statistic_from_batches(baseline_batch, batch_alpha)
        p_val_mk = mmd_test_mk.p_value_from_batches(baseline_batch, batch_alpha)
        alpha_results_mk.append(
            {"alpha": alpha_diag, "mmd": mmd_val_mk, "p_value": p_val_mk}
        )

        # SigKernel
        # mmd_val_sig = mmd_test_sig.statistic_from_batches(baseline_batch, batch_alpha)
        # p_val_sig = mmd_test_sig.p_value_from_batches(baseline_batch, batch_alpha)
        # alpha_results_sig.append(
        #     {"alpha": alpha_diag, "mmd": mmd_val_sig, "p_value": p_val_sig}
        # )

        print(f"alpha={alpha_diag:.2f}")
        print(f"  MKernel  : MMD²={mmd_val_mk:.6f}, p-value={p_val_mk:.4f}")
        # print(f"  SigKernel: MMD²={mmd_val_sig:.6f}, p-value={p_val_sig:.4f}")

    return alpha_results_mk, []  # alpha_results_sig


# ============================================================================
# Étude paramétrique sur mu
# ============================================================================


def study_mu_variation(
    baseline_batch: Batch,
    mmd_test_mk,
    mmd_test_sig,
    dim_process: int,
    num_sim: int,
    max_len: int,
    end_time: int = 30,
):
    """Étudie la variation du paramètre mu."""
    print("\n" + "=" * 80)
    print("ÉTUDE PARAMÉTRIQUE : VARIATION DE MU")
    print("=" * 80)

    mu_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
    mu_results_mk = []
    mu_results_sig = []

    for mu_val in mu_values:
        params = dict(
            mu=[mu_val, mu_val],
            alpha=[[0.3, 0.1], [0.1, 0.3]],
            beta=[[2.0, 1.0], [1.0, 2.0]],
            dim_process=dim_process,
            start_time=0,
            end_time=end_time,
        )
        seqs = simulate_hawkes(**params, num_simulations=num_sim)
        batch_mu = sequences_to_batch(seqs, max_len=max_len)

        # MKernel
        mmd_val_mk = mmd_test_mk.statistic_from_batches(baseline_batch, batch_mu)
        p_val_mk = mmd_test_mk.p_value_from_batches(baseline_batch, batch_mu)
        mu_results_mk.append({"mu": mu_val, "mmd": mmd_val_mk, "p_value": p_val_mk})

        # SigKernel
        # mmd_val_sig = mmd_test_sig.statistic_from_batches(baseline_batch, batch_mu)
        # p_val_sig = mmd_test_sig.p_value_from_batches(baseline_batch, batch_mu)
        # mu_results_sig.append({"mu": mu_val, "mmd": mmd_val_sig, "p_value": p_val_sig})

        print(f"mu={mu_val:.2f}")
        print(f"  MKernel  : MMD²={mmd_val_mk:.6f}, p-value={p_val_mk:.4f}")
        # print(f"  SigKernel: MMD²={mmd_val_sig:.6f}, p-value={p_val_sig:.4f}")

    return mu_results_mk, []  # mu_results_sig


# ============================================================================
# Génération des rapports CSV
# ============================================================================


def save_results_to_csv(
    result_same_mk,
    result_same_sig,
    results_mk,
    results_sig,
    alpha_results_mk,
    alpha_results_sig,
    mu_results_mk,
    mu_results_sig,
    output_dir: Path,
    args,
):
    """Sauvegarde tous les résultats en fichiers CSV."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Sanity check
    sanity_data = [
        {
            "test": "baseline_vs_baseline_copy",
            "kernel": "MKernel",
            "mean_mmd": result_same_mk["mean_mmd"],
            "mean_p_value": result_same_mk["mean_p_value"],
            "h0_rejected": result_same_mk["mean_p_value"] < 0.05,
            "status": "FAIL" if result_same_mk["mean_p_value"] < 0.05 else "PASS",
        },
        # {
        #     "test": "baseline_vs_baseline_copy",
        #     "kernel": "SIGKernel",
        #     "mean_mmd": result_same_sig["mean_mmd"],
        #     "mean_p_value": result_same_sig["mean_p_value"],
        #     "h0_rejected": result_same_sig["mean_p_value"] < 0.05,
        #     "status": "FAIL" if result_same_sig["mean_p_value"] < 0.05 else "PASS",
        # },
    ]
    df_sanity = pd.DataFrame(sanity_data)
    sanity_file = output_dir / f"sanity_check_{timestamp}.csv"
    df_sanity.to_csv(sanity_file, index=False)
    print(f"\n✓ Sanity check sauvegardé : {sanity_file}")
    
    # 2. Pairwise tests
    pairwise_data = []
    for (name_a, name_b), res_mk in results_mk.items():
        # res_sig = results_sig[(name_a, name_b)]
        pairwise_data.append({
            "config_a": name_a,
            "config_b": name_b,
            "kernel": "MKernel",
            "mean_mmd": res_mk["mean_mmd"],
            "mean_p_value": res_mk["mean_p_value"],
            "h0_rejected": res_mk["mean_p_value"] < 0.05,
        })
        # pairwise_data.append({
        #     "config_a": name_a,
        #     "config_b": name_b,
        #     "kernel": "SIGKernel",
        #     "mean_mmd": res_sig["mean_mmd"],
        #     "mean_p_value": res_sig["mean_p_value"],
        #     "h0_rejected": res_sig["mean_p_value"] < 0.05,
        # })
    df_pairwise = pd.DataFrame(pairwise_data)
    pairwise_file = output_dir / f"pairwise_tests_{timestamp}.csv"
    df_pairwise.to_csv(pairwise_file, index=False)
    print(f"✓ Tests par paires sauvegardés : {pairwise_file}")
    
    # 3. Parametric studies
    if alpha_results_mk:
        parametric_data = []
        for r in alpha_results_mk:
            parametric_data.append({
                "study": "alpha_variation",
                "parameter": "alpha",
                "value": r["alpha"],
                "kernel": "MKernel",
                "mmd": r["mmd"],
                "p_value": r["p_value"],
                "h0_rejected": r["p_value"] < 0.05,
            })
        # for r in alpha_results_sig:
        #     parametric_data.append({
        #         "study": "alpha_variation",
        #         "parameter": "alpha",
        #         "value": r["alpha"],
        #         "kernel": "SIGKernel",
        #         "mmd": r["mmd"],
        #         "p_value": r["p_value"],
        #         "h0_rejected": r["p_value"] < 0.05,
        #     })
        for r in mu_results_mk:
            parametric_data.append({
                "study": "mu_variation",
                "parameter": "mu",
                "value": r["mu"],
                "kernel": "MKernel",
                "mmd": r["mmd"],
                "p_value": r["p_value"],
                "h0_rejected": r["p_value"] < 0.05,
            })
        # for r in mu_results_sig:
        #     parametric_data.append({
        #         "study": "mu_variation",
        #         "parameter": "mu",
        #         "value": r["mu"],
        #         "kernel": "SIGKernel",
        #         "mmd": r["mmd"],
        #         "p_value": r["p_value"],
        #         "h0_rejected": r["p_value"] < 0.05,
        #     })
        df_parametric = pd.DataFrame(parametric_data)
        parametric_file = output_dir / f"parametric_studies_{timestamp}.csv"
        df_parametric.to_csv(parametric_file, index=False)
        print(f"✓ Études paramétriques sauvegardées : {parametric_file}")
    
    # 4. Summary metadata
    summary_data = {
        "timestamp": timestamp,
        "num_simulations": args.num_sim,
        "n_permutations": args.n_permutations,
        "batch_size": args.batch_size,
        "dim_process": args.dim_process,
        "end_time": args.end_time,
        "sanity_check_mkernel_pass": result_same_mk["mean_p_value"] > 0.05,
        # "sanity_check_sigkernel_pass": result_same_sig["mean_p_value"] > 0.05,
        "pairwise_tests_count": len(results_mk),
        "mkernel_rejections": sum(1 for r in results_mk.values() if r["mean_p_value"] < 0.05),
        # "sigkernel_rejections": sum(1 for r in results_sig.values() if r["mean_p_value"] < 0.05),
    }
    df_summary = pd.DataFrame([summary_data])
    summary_file = output_dir / f"summary_{timestamp}.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Résumé sauvegardé : {summary_file}")
    
    return output_dir


# ============================================================================
# Fonction principale
# ============================================================================


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Test MMD sur processus de Hawkes")
    parser.add_argument(
        "--num-sim", type=int, default=200, help="Nombre de simulations par config"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Nombre de permutations pour le test MMD",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Taille des batches pour les tests"
    )
    parser.add_argument(
        "--dim-process",
        type=int,
        default=2,
        help="Dimension du processus (nombre de types)",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=10,
        help="End time for Hawkes simulation (controls sequence length, default=10)",
    )
    parser.add_argument(
        "--no-parametric",
        action="store_true",
        help="Désactiver les études paramétriques",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mmd_test_results",
        help="Répertoire de sortie pour les rapports CSV",
    )

    args = parser.parse_args()

    print("=" * 90)
    print("TEST MMD SUR PROCESSUS DE HAWKES")
    print("=" * 90)
    print(f"\nParamètres:")
    print(f"  - Nombre de simulations par config : {args.num_sim}")
    print(f"  - Permutations MMD : {args.n_permutations}")
    print(f"  - Batch size : {args.batch_size}")
    print(f"  - Dimension du processus : {args.dim_process}")
    print(f"  - End time (durée simulation) : {args.end_time}")

    # Configuration
    configs = get_hawkes_configs(dim_process=args.dim_process, end_time=args.end_time)
    print(f"\n{len(configs)} configurations définies : {list(configs.keys())}")

    # Génération des données
    batches, global_max_len = generate_all_configs(configs, args.num_sim)
    
    # Memory estimation and safety check
    est_mem = estimate_memory_gb(args.batch_size, global_max_len)
    print(f"\nEstimated peak memory per kernel call: {est_mem:.1f} GB")
    if est_mem > 10:
        print(f"⚠️  WARNING: Very high memory usage expected ({est_mem:.0f} GB)!")
        print(f"   The process will likely be OOM-killed.")
        print(f"   Reduce --end-time (current: {args.end_time}) or --batch-size (current: {args.batch_size})")
        print(f"   Suggested: --end-time 10 --batch-size 16")
        return

    # Configuration des kernels
    mmd_test_mk, mmd_test_sig = setup_kernels(args.dim_process, args.n_permutations)

    # Sanity check
    result_same_mk, result_same_sig = run_sanity_check(
        batches, mmd_test_mk, mmd_test_sig, args.batch_size
    )

    # Tests entre paires
    results_mk, results_sig = run_pairwise_tests(
        configs, batches, mmd_test_mk, mmd_test_sig, args.batch_size
    )

    # Études paramétriques
    if not args.no_parametric:
        baseline_batch = batches["baseline"]

        alpha_results_mk, alpha_results_sig = study_alpha_variation(
            baseline_batch,
            mmd_test_mk,
            mmd_test_sig,
            args.dim_process,
            args.num_sim,
            global_max_len,
            end_time=args.end_time,
        )

        mu_results_mk, mu_results_sig = study_mu_variation(
            baseline_batch,
            mmd_test_mk,
            mmd_test_sig,
            args.dim_process,
            args.num_sim,
            global_max_len,
            end_time=args.end_time,
        )
    else:
        alpha_results_mk = alpha_results_sig = []
        mu_results_mk = mu_results_sig = []

    # Sauvegarde des résultats en CSV
    output_dir = Path(args.output_dir)
    save_results_to_csv(
        result_same_mk,
        result_same_sig,
        results_mk,
        results_sig,
        alpha_results_mk,
        alpha_results_sig,
        mu_results_mk,
        mu_results_sig,
        output_dir,
        args,
    )

    print("\n✅ Tests terminés avec succès!")
    print(f"📁 Rapports disponibles dans : {output_dir.absolute()}")


if __name__ == "__main__":
    main()
