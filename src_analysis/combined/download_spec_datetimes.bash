#!/bin/bash

python3 prg_geturl2.py --download_process='download_from_name' \
    --outputdir=$(pwd) \
    --processors=7 \
    --keyword='MOD35' \
    --input_csv="/scratch/midway2/koenig1/clouds/src_analysis/work_hdfs/datetime_example.csv"
