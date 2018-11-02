#!/bin/bash
#COBALT --jobname mod2cnn1
#COBALT --time 01:00:00
#COBALT --nodecount 8
#COBALT --project CSC249ADCD01
#COBALT --outputprefix /projects/CSC249ADCD01/clouds2/output/mod2cnn1/logs
#COBALT --cwd /home/casper/clouds
#COBALT --attrs pubnet


NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=$((NODES * 2))

PROJ="/projects/CSC249ADCD01/clouds2"
MODEL_PATH=$PROJ"/output/mod2cnn1"
DATA_DIR=$PROJ"/mod02tfr"

source activate clouds

echo cuda-visible-devices '"'$CUDA_VISIBLE_DEVICES'"'

mpirun -f $COBALT_NODEFILE -n $PROCS \
    python reproduction/train.py $MODEL_PATH \
        --data $DATA_DIR/"*".tfrecord \
        --shape 128 128 23 \
        --shuffle_buffer_size 1000 \
        --image_loss_weights 1 1 1 0 \
        --channel_order channels_last \
        --prefetch 8 \
        --read_threads 16 \
        --n_blocks 4 \
        --block_len 0 \
        --batch_size 128 \
        --dense_ae 256 \
        --base_dim 16 \
        --max_steps 1000 \
        --summary_every 25 \
        --save_every 50 \
