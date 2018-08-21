#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=16000

export KERAS_BACKEND=tensorflow

module purge
module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/8.0
source activate clouds_casper

MODEL_PATH=foo
DATA_DIR=/Users/casperneo/work/cloud-research/clouds
HDF_DATA=$DATA_DIR/ex.tfrecord
HDF_META=$DATA_DIR/ex.json
FIELDS='Cloud_Optical_Thickness Cloud_Effective_Radius'


rm -rf $MODEL_PATH

python dnn/train2.py $MODEL_PATH \
    --data $HDF_DATA \
    --hdf_fields $FIELDS \
    --meta_json $HDF_META \
    --epochs 1 \
    --steps_per_epoch 100 \
    --summary_every 25 \
    --n_layers 2 \
    --red_bands 0 \
    --blue_bands 0 \
    --green_bands 0 \
