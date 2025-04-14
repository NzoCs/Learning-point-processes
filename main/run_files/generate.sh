#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=resultat_%j.out
#SBATCH --error=erreur_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00  
#SBATCH --mem=40G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1      


# Activez l'environnement virtuel
cd /gpfs/users/regnaguen/Learning-point-processes/train_exp/synthetic_data_gen

source /gpfs/workdir/regnaguen/LTPP/bin/activate

python3 synthetic_data_gen.py --experiment_id hawkes1
python3 synthetic_data_gen.py --experiment_id self_correcting
python3 synthetic_data_gen.py --experiment_id H2expc
python3 synthetic_data_gen.py --experiment_id H2expi