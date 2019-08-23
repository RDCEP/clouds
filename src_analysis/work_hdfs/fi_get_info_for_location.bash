#!/bin/bash

python3 find_invalids.py --process='get_info_for_location' \
    --mod02dir='/home/koenig1/scratch-midway2/MOD021KM' \
    --mod35_dir='/home/koenig1/scratch-midway2/MOD35_L2' \
    --mod03_dir='/home/koenig1/scratch-midway2/MOD03' \
    --outputfile='output.csv' \
    --processors=4