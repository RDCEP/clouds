#!/bin/sh
#SBATCH --job-name=mod9cnn18-1km
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --time 3:00:00
#SBATCH --exclusive

module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA='/project/foster/clouds/data/mod9-724-1km/*.tfrecord'


python reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --base_dim 16 \
    --epochs 100 \
    --steps_per_epoch 5000 \
    --shape 512 512 7 \
    --n_blocks 5 \
    --block_len 1 \
    --batch_size 8 \
    --batchnorm \
    --read_threads 16 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
