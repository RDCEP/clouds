#!/bin/bash

python3 prg_geturl2.py --days=1 \
   --start='2000-02-24'\
   --end='2019-03-27'\
   --outputdir='/home/koenig1/clouds/src_analysis/combined/test' \
   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM" \
   --keyward='MOD02' \
   --thresval=1 \
   --processors=7 \
   --datedata='issuedates.txt'\
