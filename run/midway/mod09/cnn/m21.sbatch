#!/bin/sh
#SBATCH --job-name=mod9cnn21-dense-b
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --account=pi-chard

module purge
module load horovod
#module load Anaconda3/5.0.0.1
#module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA=/project/foster/clouds/data/gee/preprocessed/"*".tfrecord


python reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --base_dim 8 \
    --max_steps 150000 \
    --save_every 2500 \
    --shape 128 128 7 \
    --autoencoder_adam 0.001 0.9 0.999 \
    --n_blocks 4 \
    --block_len 0 \
    --batchnorm \
    --dense_ae 1024 \
    --read_threads 16 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
    --no_augment_rotate