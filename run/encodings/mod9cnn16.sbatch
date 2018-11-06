#!/bin/sh
#SBATCH --job-name m16enc
#SBATCH --output %x_%A.out
#SBATCH --partition gpu2
#SBATCH --account pi-chard

module purge
module load Anaconda3

source activate clouds

PFC=/project/foster/clouds
DATA=$PFC/data/gee/preprocessed/'*'.tfrecord
ENCODER=$PFC/output/mod09cnn16
OUT=$PFC/$SLURM_JOB_NAME

python experimental/read_to_bin.py \
    $OUT $ENCODER \
    --data $DATA \
    --no_augment_rotate
