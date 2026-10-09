#!/bin/bash
#SBATCH --job-name=ltpp_smoke
#SBATCH --output=err_logs/smoke_gpu_%j.out
#SBATCH --error=err_logs/smoke_gpu_%j.err
#SBATCH --partition=gpua100
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail
repo_dir=${LTPP_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
source "$repo_dir/scripts/bash/ruche_common.sh"
experiments=(NHP)
datasets=(test)
LTPP_PHASE=train
LTPP_EPOCHS=1
ltpp_launch cuda "$@"
