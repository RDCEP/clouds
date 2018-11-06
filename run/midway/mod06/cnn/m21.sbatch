#!/bin/sh
#SBATCH --job-name=m21a
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --mem=16000
#SBATCH --exclusive

export KERAS_BACKEND=tensorflow

module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
source activate clouds


MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
#DATA_DIR=/project/foster/clouds/data/nasa/mod06_l2
DATA_DIR=$SCRATCH/data/nasa/mod06_l2/
HDF_DATA=$DATA_DIR/2017_*/*.tfrecord
HDF_META=$DATA_DIR/2017_01_01/MOD06_L2.A2017001.0625.061.2017312163945.json
FIELDS='Cloud_Optical_Thickness Cloud_Water_Path Cloud_Effective_Radius'


python reproduction/train.py $MODEL_PATH \
    --data $HDF_DATA \
    --fields $FIELDS \
    --base_dim 8 \
    --meta_json $HDF_META \
    --epochs 2 \
    --n_layers 5 \
    --shape 128 128 \
    --batchnorm \
    --shuffle_buffer_size 200 \
    --red_bands 0 \
    --blue_bands 1 \
    --green_bands 2 \
    --batch_size 8
