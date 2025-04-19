#!/bin/bash
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%4

# Nettoie l'environnement module pour éviter les conflits
module purge

# Active l'environnement virtuel Python (créé avec python -m venv)
source /gpfs/workdir/regnaguen/LTPP/bin/activate

# Définition des combinaisons exp/dataset
experiments=(RMTPP_train AttNHP_train SAHP_train ODETPP_train)
datasets=(hawkes1 hawkes2_ H2expc H2expi self_correcting taxi taobao)

# Mapping index → combinaison
idx=$SLURM_ARRAY_TASK_ID
exp=${experiments[$(( idx / ${#datasets[@]} ))]}
data=${datasets[$(( idx % ${#datasets[@]} ))]}

# Lancement avec srun
srun python train.py --experiment_id "${exp}" --dataset_id "${data}"