#!/bin/bash
#SBATCH --job-name=simul
#SBATCH --output=resultat_%j.out
#SBATCH --error=erreur_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00  
#SBATCH --mem=20G
#SBATCH --partition=cpu_long    # Ajoutez cette ligne (remplacez "cpu_long" par le nom réel de la partition)

# Activez l'environnement virtuel
cd /gpfs/users/regnaguen/ETPP_modifs/train_experiment

source /gpfs/workdir/regnaguen/LTPP/bin/activate

python3 simulation.py 