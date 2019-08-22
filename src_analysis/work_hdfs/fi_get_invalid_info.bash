#!/bin/bash

python3 prg_geturl2.py --process='get_invalid_info' \
	--input='clustering_invalid_filelists.txt' \
    --mod02dir='/home/koenig1/scratch-midway2/MOD021KM' \
    --mod35_dir='/home/koenig1/scratch-midway2/MOD35_L2' \
    --mod03_dir='/home/koenig1/scratch-midway2/MOD03' \
    --outputfile='output.csv' \
    --processors=4