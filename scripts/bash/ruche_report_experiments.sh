#!/usr/bin/env bash
# Opt-in only. Supply account/partition/GPU/time/array limits with sbatch.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
environment=${UV_PROJECT_ENVIRONMENT:-$repo_dir/.venv}
python=$environment/bin/python
output=${LTPP_SCIENCE_OUTPUT:-$repo_dir/artifacts/scientific-report}
cases=(poisson_H0 hawkes_exp_H0 hawkes_power_H0 poisson_timing_H0
       poisson_rate_110 hawkes_alpha_8 exp_vs_power_p2 exp_vs_power_p3
       hawkes_exp_timing hawkes_power_timing)
dry_run=false
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    dry_run=true
elif (( $# != 0 )); then
    echo "Usage: bash scripts/bash/ruche_report_experiments.sh [--dry-run]" >&2
    exit 2
fi
if $dry_run; then
    first=0; last=19
else
    if [[ -z ${SLURM_JOB_ID:-} || ! ${SLURM_ARRAY_TASK_ID:-} =~ ^([0-9]|1[0-9])$ ]]; then
        echo "Submit as a Slurm array 0-19; no execution on the login node." >&2
        exit 2
    fi
    if [[ ! -x $python || ! -f $repo_dir/uv.lock ]]; then
        echo "Install the locked Ruche environment before submitting." >&2
        exit 2
    fi
    first=$SLURM_ARRAY_TASK_ID; last=$first
fi
cd -- "$repo_dir"
for (( index=first; index<=last; index++ )); do
    case_name=${cases[$((index / 2))]}
    cohort=$((index % 2))
    trials=250
    if [[ $case_name == *H0 ]]; then trials=1000; fi
    command=("$python" -m scripts.report_experiment_validation
             --case "$case_name" --cohort "$cohort" --trials "$trials"
             --n 128 --permutations 499 --grid 100 --device cuda
             --output "$output/$case_name/cohort-$cohort")
    if $dry_run; then
        printf 'task=%s ' "$index"; printf '%q ' "${command[@]}"; printf '\n'
    else
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        export MPLBACKEND=Agg WANDB_MODE=offline HF_HUB_OFFLINE=1
        srun "$python" -m scripts.check_backend --device cuda
        srun "${command[@]}"
    fi
done
