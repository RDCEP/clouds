#!/bin/sh
#SBATCH --job-name to_bin
#SBATCH --output %x_%A.out
#SBATCH --partition gpu2
#SBATCH --account pi-chard

module purge
module load Anaconda3

source activate clouds

PFC=/project/foster/clouds
DATA=$PFC/data/gee/preprocessed/'*'.tfrecord
ENCODER=$PFC/output/mod09cnn17
OUT=$PFC/mod09cnn17-encodings

python experimental/read_to_bin.py \
    $OUT $ENCODER \
    --data $DATA \
    --no_augment_rotate
