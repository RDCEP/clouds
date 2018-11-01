#!/bin/bash
#COBALT --jobname translate-mod021
#COBALT --time 01:00:00
#COBALT --nodecount 128
#COBALT --outputprefix theta-logs/translate-mod021
#COBALT --cwd /home/casper/clouds
#COBALT --attrs=pubnet


source activate clouds


DATA='/projects/CSC249ADCD01/clouds2/mod021km'
OUT='/projects/CSC249ADCD01/clouds2/mod2-tfr'
MODE='mod02_1km'
STRIDES=64 64
SHAPE="182 182"

mp
python3 reproduction/pipeline/into_record.py \
		$DATA $OUT $MODE \
		--stride $STRIDES \
 		--shape $SHAPE \
