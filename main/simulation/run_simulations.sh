#!/bin/bash
#SBATCH --output=err_logs/simul_%A_%a.out
#BATCH --error=err_logs/simul_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-62%4

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l'environnement virtuel Python (créé avec python -m venv)
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des combinaisons modèle/dataset
models=(NHP_simul RMTPP AttNHP SAHP THP FullyNN IntensityFree)
datasets=(hawkes1 H2expc H2expi self_correcting taxi taobao hawkes2 hawkes2_)

# Création du dossier pour les logs d'erreurs s'il n'existe pas déjà
mkdir -p err_logs

# Mapping index → combinaison
idx=$SLURM_ARRAY_TASK_ID
model=${models[$(( idx / ${#datasets[@]} ))]}
dataset=${datasets[$(( idx % ${#datasets[@]} ))]}

# Définition du chemin du modèle pour la combinaison actuelle
model_dir="../train/checkpoints/${model}/trained_models/${dataset}"

# Vérification de l'existence du modèle entraîné
if [ "${model_dir}/best.ckpt" ]; then
    echo "Lancement de la simulation pour ${model} sur ${dataset}"
    # Lancement avec srun
    srun python run_simulation.py --experiment_id "${model}" --dataset_id "${dataset}"
else
    echo "Le fichier ${model_dir}/best.ckpt n'existe pas. Simulation ignorée pour ${model} sur ${dataset}."
fi
