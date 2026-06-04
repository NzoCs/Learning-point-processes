#!/bin/bash
#
#SBATCH --job-name=hawkes_analysis
#SBATCH --output=/raid/home/students/regna_enz/Learning-point-processes/out.out
#SBATCH --error=/raid/home/students/regna_enz/Learning-point-processes/out.out

## Mails
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%VOTRE_EMAIL%

#SBATCH --partition=prod40

## gpu:nvidia_a100_1g.10gb:1 pour prod10
## gpu:nvidia_a100_3g.40gb:1 pour prod40
## gpu:nvidia_a100-sxm4-80gb:1

#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1

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
new-ltpp run --phase all --general-specs-config h64 --training-config e500_b1 --thinning-config e200_s60 --model NHP --data-loading-config b64_w4 --dataset-id hawkes1 --statistical-test-config sig_kernel
