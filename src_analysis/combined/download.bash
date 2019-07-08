#!/bin/bash

python3 prg_geturl2.py --days=1 \
   --start='2000-02-24'\
   --end='2019-03-27'\
   --outputdir='/home/koenig1/scratch-midway2/MOD35/clustering/specific_dates' \
   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2" \
   --keyward='MOD35_L2.A' \
   --thresval=1 \
   --processors=7 \
   --datedata='clustering_mod35_list.txt'\

