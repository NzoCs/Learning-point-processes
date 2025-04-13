#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=resultat_%j.out
#SBATCH --error=erreur_%j.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00  
#SBATCH --mem=40G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:2      # Demande 2 GPUs au total

# Exécutez les scripts en parallèle, chacun avec 1 GPU

# Activez l'environnement virtuel
cd /gpfs/users/regnaguen/ETPP_modifs/train_experiment

source /gpfs/workdir/regnaguen/LTPP/bin/activate

python3 train.py --experiment_id NHP_hawkes1_train