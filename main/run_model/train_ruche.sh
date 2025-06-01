#!/bin/bash
#SBATCH --job-name=train_gpu
#SBATCH --output=err_logs/train_gpu_%A_%a.out
#SBATCH --error=err_logs/train_gpu_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --array=0-47%5

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l'environnement virtuel Python
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des combinaisons exp/dataset
experiments=(NHP THP IntensityFree SAHP)
datasets=(hawkes1 H2expc H2expi self_correcting hawkes2 taxi taobao amazon)

# Mapping index → combinaison
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$(( idx / ${#datasets[@]} ))]}
data=${datasets[$(( idx % ${#datasets[@]} ))]}

# Lancement avec srun
srun python run.py --experiment_id "${exp}" --dataset_id "${data}" --phase "all"