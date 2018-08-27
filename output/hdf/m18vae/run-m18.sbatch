#!/bin/sh
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --mem=16000

export KERAS_BACKEND=tensorflow

module purge
module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/9.0
source activate clouds


MODEL_PATH=out/hdf/m18vae
DATA_DIR=/project/foster/clouds/nasa_laads/mod06_l2/
HDF_DATA=$DATA_DIR/2017_*/*.tfrecord
HDF_META=$DATA_DIR/2017_01_01/MOD06_L2.A2017001.0625.061.2017312163945.json
FIELDS='Cloud_Optical_Thickness Cloud_Water_Path Cirrus_Reflectance'

python train2.py $MODEL_PATH \
    --data $HDF_DATA \
    --hdf_fields $FIELDS \
    --meta_json $HDF_META \
    --variational \
    --epochs 500 \
    --n_layers 3 \
    --red_bands 0 \
    --blue_bands 1 \
    --green_bands 2 \
