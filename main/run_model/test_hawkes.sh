#!/bin/bash
#SBATCH --job-name=test_hawkes
#SBATCH --output=err_logs/test_hawkes_%A_%a.out
#SBATCH --error=err_logs/test_hawkes_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --partition=mem
#SBATCH --array=0-3%5

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l’environnement virtuel Python
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des expériences
experiments=(H2expc H2expi hawkes2 hawkes1)

# Mapping SLURM task index to experiment and dataset
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$idx]}
data=${exp}

# Lancement avec srun
srun python run.py --experiment_id "${exp}" --dataset_id "${data}" --phase "test"
