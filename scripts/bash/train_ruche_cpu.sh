#!/bin/bash
#SBATCH --job-name=train_cpu
#SBATCH --output=err_logs/train_cpu_%A_%a.out
#SBATCH --error=err_logs/train_cpu_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --partition=cpu_long
#SBATCH --array=0-27%5

set -euo pipefail
repo_dir=${LTPP_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
source "$repo_dir/scripts/bash/ruche_common.sh"
experiments=(NHP THP IntensityFree SAHP)
# self_correcting has no dataset preset in the delivered configuration.
datasets=(hawkes1 H2expc H2expi hawkes2 taxi taobao amazon)
ltpp_launch cpu "$@"
