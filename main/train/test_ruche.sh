#!/bin/bash
#SBATCH --output=err_logs/test_%A_%a.out
#SBATCH --error=err_logs/test_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-24%5

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l’environnement virtuel Python (créé avec python -m venv)
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des combinaisons exp/dataset
experiments=(NHP_test RMTPP_test AttNHP_test SAHP_test)
datasets=(hawkes1 H2expc H2expi self_correcting taxi taobao self_correcting hawkes2_)

# Mapping index → combinaison
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$(( idx / ${#datasets[@]} ))]}
data=${datasets[$(( idx % ${#datasets[@]} ))]}

# Définition du chemin du modèle pour la combinaison actuelle
model_dir="./checkpoints/${exp}/trained_models/${data}"

# Vérification de l'existence de best.ckpt et l'absence de test_results.json
if [ -f "${model_dir}/best.ckpt" ] && [ ! -f "${model_dir}/test_results.json" ]; then
    echo "Lancement du test pour ${exp} sur ${data}"
# Lancement avec srun
srun python test.py --experiment_id "${exp}" --dataset_id "${data}"
else
    if [ ! -f "${model_dir}/best.ckpt" ]; then
        echo "Le fichier ${model_dir}/best.ckpt n'existe pas. Test ignoré."
    fi
    if [ -f "${model_dir}/test_results.json" ]; then
        echo "Le fichier ${model_dir}/test_results.json existe déjà. Test ignoré."
    fi
fi