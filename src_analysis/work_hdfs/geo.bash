#!/bin/bash

python3 geolocation.py	--input_file='/home/koenig1/clouds/src_analysis/combined/mod02_files_big_patches.csv' \
    --mod02dir='/home/koenig1/scratch-midway2/big_invalids/mod02invalids' \
    --mod35_dir='/home/koenig1/scratch-midway2/big_invalids/mod35invalids' \
    --mod03_dir='/home/koenig1/scratch-midway2/big_invalids/mod03invalids' \
    --processors=7 \
    --outputfile='invalids_output.csv' \
