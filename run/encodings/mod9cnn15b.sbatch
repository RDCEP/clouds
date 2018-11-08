#!/bin/sh
#SBATCH --job-name m15b-enc
#SBATCH --output %x_%A.out
#SBATCH --partition gpu2
#SBATCH --account pi-chard

module purge
module load Anaconda3
#module load cuda/9.0

source activate clouds
#source activate cc2


PFC=/project/foster/clouds
DATA=$PFC/data/gee/preprocessed/'*'.tfrecord
ENCODER=$PFC/output/mod09cnn15b
STEP=380000
OUT=$PFC/$SLURM_JOB_NAME

python experimental/read_to_bin.py \
    $OUT $ENCODER $STEP \
    --data $DATA \
    --no_augment_rotate
