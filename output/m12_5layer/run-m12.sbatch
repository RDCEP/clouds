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

DATA=/project/foster/clouds/geotfr/*.tfrecords
MODEL_PATH=out/m12_5layer

python train2.py $MODEL_PATH --data $DATA --epochs 500 --n_layers=5
