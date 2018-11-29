#!/bin/bash
#COBALT --jobname m9cnnm21
#COBALT --outputprefix theta-logs/$COBALT_JOBNAME-debug
#COBALT --time 00:30:00
#COBALT --nodecount 4
#COBALT --project CSC249ADCD01
#COBALT --queue debug-cache-quad
#COBALT --cwd /home/rlourenc/rdcep_clouds

PROJ='/projects/CSC249ADCD01/clouds'
MODEL_PATH=$PROJ'/output/$COBALT_JOBNAME'-debug'
DATA_DIR=$PROJ'/data/mod09/2015_05'

module load datascience/tensorflow-1.10
module load datascience/horovod-0.15.0

### n = number of processes ; N = no. proc / node
aprun -n 64 -N 16 \
    python3 reproduction/train.py $MODEL_PATH \
        --data $DATA_DIR/"*".tfrecord \
        --shape 128 128 7 \
        --batch_size 8 \
        --max_steps 150000 \
        --save_every 2500 \
        --summary_every 250 \
        --autoencoder_adam 0.001 0.9 0.999\
        --n_blocks 4 \
        --base_dim 8 \
        --block_len 0 \
        --batchnorm \
        --read_threads 16 \
        --shuffle_buffer_size 1000 \
        --image_loss_weights 1 1 1 1 \
        --no_augment_rotate