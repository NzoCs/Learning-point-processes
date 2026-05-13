#!/bin/bash
#
#SBATCH --job-name=hawkes_analysis
#SBATCH --output=/raid/home/students/regna_enz/Learning-point-processes/out.out
#SBATCH --error=/raid/home/students/regna_enz/Learning-point-processes/out.out

## Mails
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%VOTRE_EMAIL%

#SBATCH --partition=prod10

## 1g.10gb:1 pour prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1

## total requested cpus (ntasks * cpus-per-task) must be in [1: 4 * nb_1g.10gb]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:0:0

## Activer l'environnement virtuel
source /raid/home/students/regna_enz/SignatureMMDTesting/.venv/bin/activate

## Aller dans le répertoire de travail
cd /raid/home/students/regna_enz/Learning-point-processes
source .venv/bin/activate

## Lancer le script
export PYTHONUNBUFFERED=1
new-ltpp run --model THP --data-config taxi --phase all --epochs 50
