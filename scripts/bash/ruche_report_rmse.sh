#!/usr/bin/env bash
# Manual Slurm array 0-19, preferably --dependency=aftercorr:<MMD-array>.
set -euo pipefail
: "${LTPP_MMD_SOURCE:?Set the completed scientific campaign cells directory}"
: "${LTPP_RMSE_OUTPUT:?Set a separate RMSE output directory}"
: "${UV_PROJECT_ENVIRONMENT:?Set the installed locked environment}"
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?Submit array 0-19}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] && (( SLURM_ARRAY_TASK_ID < 20 )) || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python="$UV_PROJECT_ENVIRONMENT/bin/python"
cases=(poisson_H0 hawkes_exp_H0 hawkes_power_H0 poisson_timing_H0 poisson_rate_110 hawkes_alpha_8 exp_vs_power_p2 exp_vs_power_p3 hawkes_exp_timing hawkes_power_timing)
case_name="${cases[$((SLURM_ARRAY_TASK_ID / 2))]}"
cohort="$((SLURM_ARRAY_TASK_ID % 2))"
srun "$python" -m scripts.report_rmse_validation replay \
  "$LTPP_MMD_SOURCE/$case_name/cohort-$cohort" \
  "$LTPP_RMSE_OUTPUT/$case_name/cohort-$cohort" --descriptive
