#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=resultat_%j.out
#SBATCH --error=erreur_%j.err
#SBATCH --ntasks=1
#SBATCH --time=50:00:00  
#SBATCH --mem=100G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:5      

# Exécutez les scripts en parallèle, chacun avec 1 GPU

# Activez l'environnement virtuel
cd /gpfs/users/regnaguen/New_LTPP/main/train

source /gpfs/workdir/regnaguen/LTPP/bin/activate


python train.py --experiment_id NHP_train --dataset_id hawkes1
python train.py --experiment_id NHP_train --dataset_id hawkes2
python train.py --experiment_id NHP_train --dataset_id H2expc
python train.py --experiment_id NHP_train --dataset_id H2expi
python train.py --experiment_id NHP_train --dataset_id self_correcting
python train.py --experiment_id RMTPP_train --dataset_id hawkes1
python train.py --experiment_id RMTPP_train --dataset_id hawkes2
python train.py --experiment_id RMTPP_train --dataset_id H2expc
python train.py --experiment_id RMTPP_train --dataset_id H2expi
python train.py --experiment_id RMTPP_train --dataset_id self_correcting
python train.py --experiment_id AttNHP_train --dataset_id hawkes1
python train.py --experiment_id AttNHP_train --dataset_id hawkes2
python train.py --experiment_id AttNHP_train --dataset_id H2expc
python train.py --experiment_id AttNHP_train --dataset_id H2expi
python train.py --experiment_id AttNHP_train --dataset_id self_correcting