#!/usr/bin/env bash
# Explicit opt-in; requires frozen training and a successful six-reference pilot.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python="${UV_PROJECT_ENVIRONMENT:-$repo/.venv}/bin/python"
: "${LTPP_KNOWN_RMSE_SOURCE:?Set completed known-reference all-event cells}"
: "${LTPP_LEARNED_TRAINING:?Set the complete frozen training directory}"
: "${LTPP_LEARNED_OUTPUT:?Set a new learned-reference campaign directory}"
dry_run=false
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
  dry_run=true; first=0; last=19
elif (( $# == 0 )) && [[ -n ${SLURM_JOB_ID:-} && ${SLURM_ARRAY_TASK_ID:-} =~ ^([0-9]|1[0-9])$ ]]; then
  first=$SLURM_ARRAY_TASK_ID; last=$first
else
  echo "Use --dry-run or a Slurm array 0-19." >&2
  exit 2
fi
cd -- "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if ! $dry_run; then
  "$python" -c 'import json,os; r=json.load(open(os.environ["LTPP_LEARNED_OUTPUT"]+"/pilot/backend.json")); assert r["status"]=="passed" and len(r["checks"])==6'
fi
cases=(poisson_H0 hawkes_exp_H0 hawkes_power_H0 poisson_timing_H0 poisson_rate_110 hawkes_alpha_8 exp_vs_power_p2 exp_vs_power_p3 hawkes_exp_timing hawkes_power_timing)
for (( index=first; index<=last; index++ )); do
  case_name=${cases[$((index / 2))]}; cohort=$((index % 2))
  command=("$python" -m scripts.report_learned_rmse replay
    "$LTPP_KNOWN_RMSE_SOURCE/$case_name/cohort-$cohort"
    "$LTPP_LEARNED_TRAINING"
    "$LTPP_LEARNED_OUTPUT/cells/$case_name/cohort-$cohort" --device cuda)
  if $dry_run; then
    printf 'task=%s ' "$index"; printf '%q ' "${command[@]}"; printf '\n'
  else
    srun "${command[@]}"
  fi
done
