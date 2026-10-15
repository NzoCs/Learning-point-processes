#!/bin/bash
#SBATCH --job-name=train_gpu
#SBATCH --output=err_logs/train_gpu_%A_%a.out
#SBATCH --error=err_logs/train_gpu_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

set -euo pipefail
repo_dir=${LTPP_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}
source "$repo_dir/scripts/bash/ruche_common.sh"
experiments=(IntensityFree)
datasets=(taobao taxi amazon)
ltpp_launch cuda "$@"
