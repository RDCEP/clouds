#!/bin/bash
#COBALT --jobname mod9cnn15
#COBALT --outputprefix theta-logs/mod9cnn15-debug
#COBALT --queue debug-cache-quad
#COBALT --cwd /home/casper/clouds
#COBALT --time 00:30:00
#COBALT --nodecount 4
#COBALT --project CSC249ADCD01
#COBALT --attrs=pubnet


PROJ='/projects/CSC249ADCD01/clouds'
MODEL_PATH=$PROJ'/output/mod9cnn15-debug'
DATA_DIR=$PROJ'/data/mod09-whitened'

module load datascience/tensorflow-1.10
module load datascience/horovod-0.15.0

aprun -n 64 -N 16 \
    python3 reproduction/train.py $MODEL_PATH \
        --data $DATA_DIR/"*".tfrecord \
        --shape 128 128 7 \
        --batch_size 8 \
        --max_steps 100000 \
        --save_every 5000 \
        --summary_every 250 \
        --autoencoder_adam 0.001 0.9 0.999\
        --n_blocks 4 \
        --base_dim 16 \
        --block_len 0 \
        --batchnorm \
        --read_threads 4 \
        --shuffle_buffer_size 1000 \
        --image_loss_weights 1 1 1 1 \
        --no_augment_rotate 
