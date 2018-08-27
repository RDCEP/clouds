
MODEL_PATH=experimental/test/foo
DATA_DIR=/Users/casperneo/work/cloud-research/clouds/experimental/test
HDF_DATA=$DATA_DIR/ex.tfrecord
HDF_META=$DATA_DIR/ex.json
FIELDS='Cloud_Optical_Thickness Cloud_Water_Path Cirrus_Reflectance'

rm -rf $MODEL_PATH

python reproduction/train.py $MODEL_PATH \
    --data $HDF_DATA \
    --fields $FIELDS \
    --meta_json $HDF_META \
    --epochs 1 \
    --steps_per_epoch 100 \
    --summary_every 25 \
    --n_layers 3 \
    --red_bands 0 \
    --blue_bands 0 \
    --green_bands 0 \
    --variational
