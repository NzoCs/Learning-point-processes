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
#SBATCH --export=NONE
#SBATCH --array=0-3%5 # Adjusted for 2 experiments (hawkes1, hawkes2)

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l’environnement virtuel Python (créé avec python -m venv)
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des expériences
# For this script, we want specific pairings:
# hawkes1 experiment with hawkes1 dataset
# hawkes2 experiment with hawkes2 dataset
experiments=(hawkes1 hawkes2 H2expc H2expi)

# Mapping SLURM task index to experiment and dataset
# The dataset name will be the same as the experiment name.
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$idx]}
data=${exp} # Ensures dataset is the same as the experiment

# Définition du chemin du modèle pour la combinaison actuelle
model_dir="./checkpoints/${exp}/trained_models/${data}"

# Vérification de l'existence de best.ckpt et l'absence de test_results.json
if [ -f "${model_dir}/best.ckpt" ] && [ ! -f "${model_dir}/test_results.json" ]; then
    echo "Lancement du test pour ${exp} sur ${data}"
# Lancement avec srun
srun python test.py --experiment_id "${exp}" --dataset_id "${data}"
else
    if [ ! -f "${model_dir}/best.ckpt" ]; then
        echo "Le fichier ${model_dir}/best.ckpt n'existe pas. Test ignoré pour ${exp} sur ${data}."
    fi
    if [ -f "${model_dir}/test_results.json" ]; then
        echo "Le fichier ${model_dir}/test_results.json existe déjà. Test ignoré pour ${exp} sur ${data}."
    fi
fi
