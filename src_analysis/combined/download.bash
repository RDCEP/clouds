#!/bin/bash

python3 prg_geturl2.py --days=1 \
   --start='2000-02-24'\
   --end='2019-03-27'\
   --outputdir='MOD35/clustering/' \
   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2" \
   --keyward='MOD35_L2.A' \
   --thresval=1 \
   --processors=7 \
   --datedata='mod02_training_dates.txt'\

