#!/bin/bash
#COBALT --jobname test-horovod
#COBALT --time 00:30:00
#COBALT --nodecount 2
#COBALT --project CSC249ADCD01
#COBALT --outputprefix test-horovod2/logs
#COBALT --cwd /home/casper/clouds
#COBALT --attrs=pubnet

MODEL_PATH='/home/casper/clouds/test-horovod2/'
DATA_DIR="/projects/CSC249ADCD01/clouds2/mod02tfr"

source activate clouds

mpiexec -n 2 \
    python reproduction/train.py $MODEL_PATH \
        --data $DATA_DIR/"*".tfrecord \
        --shape 128 128 7 \
        --shuffle_buffer_size 1000 \
        --channel_order channels_last \
        --prefetch 8 \
        --read_threads 16 \
        --n_blocks 4 \
        --block_len 0 \
        --batch_size 104 \
        --dense_ae 256 \
        --base_dim 16 \
        --max_steps 1000 \
        --summary_every 25 \
        --save_every 50 \
        --dense_ae 512 \
        --no_augment_rotate \
