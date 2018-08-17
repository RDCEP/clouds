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

MODEL_PATH=out/hdf/m16-5l
DATA_DIR=/project/foster/clouds/nasa_laads/mod06_l2/order501256920/
#HDF_DATA=$DATA_DIR/records3/*.tfrecord
HDF_DATA=$DATA_DIR/records3/MOD06_L2.A2017001.0115.061.2017312163804.tfrecord
HDF_META=$DATA_DIR/jsons3/MOD06_L2.A2017001.0115.061.2017312163804.json
#FIELDS='Cloud_Optical_Thickness Cloud_Effective_Radius'
FIELDS='Cloud_Optical_Thickness'


python train2.py $MODEL_PATH \
	--data $HDF_DATA \
	--hdf_fields $FIELDS \
	--meta_json $HDF_META \
	--epochs 500 \
   	--n_layers 5 \
    	--red_bands 0 \
	--blue_bands 0 \
	--green_bands 0 \

