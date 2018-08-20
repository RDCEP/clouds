
MODEL_PATH=foo
DATA_DIR=/Users/casperneo/work/cloud-research/clouds
HDF_DATA=$DATA_DIR/ex.tfrecord
HDF_META=$DATA_DIR/ex.json
FIELDS='Cloud_Optical_Thickness Cloud_Effective_Radius'

python dnn/train2.py $MODEL_PATH \
    --data $HDF_DATA \
    --hdf_fields $FIELDS \
    --meta_json $HDF_META \
    --epochs 500 \
    --n_layers 5 \
    --red_bands 0 \
    --blue_bands 0 \
    --green_bands 0 \
