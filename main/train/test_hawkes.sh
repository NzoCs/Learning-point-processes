#!/bin/bash
#SBATCH --output=err_logs/test_hawkes_%A_%a.out
#SBATCH --error=err_logs/test_hawkes_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%5

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l’environnement virtuel Python (créé avec python -m venv)
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des expériences
experiments=(H2expc)

# Mapping SLURM task index to experiment and dataset
# The dataset name will be the same as the experiment name.
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$idx]}
data=${exp} # Ensures dataset is the same as the experiment


# Lancement avec srun
srun python test.py --experiment_id "${exp}" --dataset_id "${data}"
