#!/bin/bash

#SBATCH --account=pi-foster
#SBATCH --job-name=label_clf
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=bigmem2
##SBATCH --partition=broadwl
##SBATCH --partition=broadwl-lc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=120000 #MB
#SBATCH --time=00:40:00

module load python/anaconda-2020.02
source activate tf-cpu

python3 label_classifier.py
