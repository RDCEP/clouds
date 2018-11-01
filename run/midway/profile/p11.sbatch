#!/bin/sh
#SBATCH --job-name=p11
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --exclusive

module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA='/project/foster/clouds/data/gee/preprocessed/*.tfrecord'


python reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --base_dim 32 \
    --epochs 1 \
    --steps_per_epoch 5000 \
    --summary_every 100 \
    --shape 128 128 7 \
    --n_blocks 5 \
    --block_len 3 \
    --batchnorm \
    --read_threads 16 \
    --shuffle_buffer_size 1000 \
