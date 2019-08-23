#!/bin/bash

python3 geolocation.py	--input_file='/home/koenig1/clouds/src_analysis/work_hdfs/mod02_geo_example.csv' \
    --mod02dir='/home/koenig1/scratch-midway2/big_invalids/mod02invalids' \
    --mod35_dir='/home/koenig1/scratch-midway2/big_invalids/mod35invalids' \
    --mod03_dir='/home/koenig1/scratch-midway2/big_invalids/mod03invalids' \
    --processors=7 \
    --outputfile='invalids_output.csv' \
