#!/bin/sh
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=16000

export KERAS_BACKEND=tensorflow

module purge
module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/9.0
source activate clouds

# Be sure to double escape * as its substituted twice before python
DATA=/project/foster/clouds/geotfr/*.tfrecords
MODEL_PATH=out/m6

python train.py $MODEL_PATH --data $DATA --epochs 200
