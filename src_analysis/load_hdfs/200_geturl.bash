#!/bin/bash

python3 prg_geturl.py \
   --datedata='/home/koenig1/clouds/src_analysis/metadata/output.txt'\
   --outputdir='/home/koenig1/clouds/src_analysis/load_hdfs/test' \
   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2" \
   --keyward='MOD35_L2.A' \
   --thresval=1 \
   --processor=1

   #--datedata='/home/tkurihana/src/src_metadata/c0.txt'\
## past parser
#1 when training data load 
#   --datedata='/home/tkurihana/src/src_metadata/mod02.txt'\
#   --outputdir=/home/tkurihana/scratch-midway2/data/MOD02/laads_2000_2018 \
#   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM" \
