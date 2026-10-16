#!/usr/bin/env bash
# Opt-in only: validate the numerical CUDA pilot before submitting array 0-19%1.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python="${UV_PROJECT_ENVIRONMENT:-$repo/.venv}/bin/python"
: "${LTPP_MMD_SOURCE:?Set the original MMD cells root}"
: "${LTPP_LANDMARK_SOURCE:?Set the completed original four-date RMSE cells root}"
: "${LTPP_ALL_EVENT_OUTPUT:?Use a separate all-event output directory}"
dry_run=false
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
  dry_run=true; first=0; last=19
elif (( $# == 0 )) && [[ -n ${SLURM_JOB_ID:-} && ${SLURM_ARRAY_TASK_ID:-} =~ ^([0-9]|1[0-9])$ ]]; then
  first=$SLURM_ARRAY_TASK_ID; last=$first
else
  echo "Use --dry-run or submit a Slurm array 0-19." >&2
  exit 2
fi
cases=(poisson_H0 hawkes_exp_H0 hawkes_power_H0 poisson_timing_H0 poisson_rate_110 hawkes_alpha_8 exp_vs_power_p2 exp_vs_power_p3 hawkes_exp_timing hawkes_power_timing)
cd -- "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for (( index=first; index<=last; index++ )); do
  case_name=${cases[$((index / 2))]}; cohort=$((index % 2))
  command=("$python" -m scripts.report_rmse_validation replay
    "$LTPP_MMD_SOURCE/$case_name/cohort-$cohort"
    "$LTPP_ALL_EVENT_OUTPUT/$case_name/cohort-$cohort"
    --mode all_events --device cuda
    --landmark-source "$LTPP_LANDMARK_SOURCE/$case_name/cohort-$cohort")
  if $dry_run; then
    printf 'task=%s ' "$index"; printf '%q ' "${command[@]}"; printf '\n'
  else
    srun "${command[@]}"
  fi
done
