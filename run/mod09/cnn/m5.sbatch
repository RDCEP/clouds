#!/bin/sh
#SBATCH --job-name=mod09cnn5
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --mem=64000
#SBATCH --exclusive

export KERAS_BACKEND=tensorflow

module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA_DIR=/project/foster/clouds/data/gee/records
RECORDS=$DATA_DIR/*.tfrecord
META=$DATA_DIR/2017-01-01_MOD09GA_background_removal_zero_inputated_image_with_cf_50perc_grid_size10-0000000000-0000070656.json
FIELDS='b1 b2 b3 b4 b5 b6 b7'

python reproduction/train.py $MODEL_PATH \
    --data $RECORDS \
    --fields $FIELDS \
    --base_dim 8 \
    --meta_json $META \
    --epochs 500 \
    --shape 256 256 \
    --n_blocks 6 \
    --block_len 2 \
    --batchnorm \
    --normalization whiten \
    --read_threads 28 \
    --shuffle_buffer_size 100
