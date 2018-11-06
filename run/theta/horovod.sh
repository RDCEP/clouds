#!/bin/bash
#COBALT --jobname test-horovod-4
#COBALT --time 00:30:00
#COBALT --nodecount 4
#COBALT --project CSC249ADCD01
#COBALT --queue debug-cache-quad
#COBALT --outputprefix test-horovod-4/logs
#COBALT --cwd /home/casper/clouds
#COBALT --attrs=pubnet

MODEL_PATH='/home/casper/clouds/test-horovod-4/'
DATA_DIR="/home/casper/tmp"

#module purge
#module load miniconda-3.6
module load datascience/tensorflow-1.10
module load datascience/horovod-0.15.0

echo which aprun
#source activate clouds

aprun -n 32 -N 8 \
    python3 reproduction/train.py $MODEL_PATH \
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
