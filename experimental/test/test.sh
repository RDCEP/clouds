
MODEL_PATH=experimental/test/foo
DATA_DIR=/Users/casperneo/work/cloud-research/clouds/data/tif2
DATA=$DATA_DIR/*.tfrecord
META=$DATA_DIR/closed-open-cell-south-pacific.json
# FIELDS='Cloud_Optical_Thickness Cloud_Water_Path Cloud_Effective_Radius'
FIELDS='b1 b2 b3 b4 b5 b6 b7'

rm -rf $MODEL_PATH

OUT=date 'test'

python reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --shape 64 64 7 \
    --shuffle_buffer_size 200 \
    --prefetch 8 \
    --n_blocks 3 \
    --block_len 1 \
    --base_dim 16 \
    --epochs 3 \
    --summary_every 25 \
    | tee test-`date +%b%d-%H:%M`.out
