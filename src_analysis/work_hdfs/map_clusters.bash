#!/bin/bash

python3 map_cluster.py --txt_file='filelist_metadata_train_random-80000_nc-20_m01_b28_29_31_patches_labels_2000-2018_random_aggl.txt' \
    --input_dir='/home/koenig1/scratch-midway2/clusters_20/group0' \
    --mod03_dir='/home/koenig1/scratch-midway2/clusters_20' \
    --num_patches=80000 \
    --output_csv='output.csv' \
    --npz_dir='/home/koenig1/scratch-midway2/clusters_20/output_clouds_feature_2000_2018_validfiles' \
    --nparts=7 \
	--map_info=['map_clusters','map_by_date'] \
