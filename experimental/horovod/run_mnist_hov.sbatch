#!/bin/sh
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=16000
#SBATCH --account=pi-foster

export KERAS_BACKEND=tensorflow

module purge
export PATH="/home/rlourenco/miniconda2/bin:$PATH"
module load cuda/9.0
source activate python3env

python3 mnist_test.py
