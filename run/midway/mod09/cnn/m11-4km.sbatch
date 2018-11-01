#!/bin/sh
#SBATCH --job-name=mod09cnn11-4km
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err


module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA='/project/foster/clouds/data/gee/4km_pxl/*.tfrecord'


python reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --base_dim 8 \
    --epochs 50 \
    --steps_per_epoch 10000 \
    --shape 64 64 7 \
    --n_blocks 5 \
    --block_len 1 \
    --batchnorm \
    --read_threads 16 \
    --shuffle_buffer_size 100
