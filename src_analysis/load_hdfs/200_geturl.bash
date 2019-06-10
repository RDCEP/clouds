#!/bin/bash

python prg_geturl.py \
   --datedata='/home/tkurihana/src/src_metadata/c0_.txt'\
   --outputdir=/home/tkurihana/scratch-midway2/data/MOD02/clustering_laads_2000_2018_5 \
   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM" \
   --keyward='MOD021KM.A' \
   --thresval=100 

   #--datedata='/home/tkurihana/src/src_metadata/c0.txt'\
## past parser
#1 when training data load 
#   --datedata='/home/tkurihana/src/src_metadata/mod02.txt'\
#   --outputdir=/home/tkurihana/scratch-midway2/data/MOD02/laads_2000_2018 \
#   --url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM" \
