#!/bin/bash
#COBALT --jobname translate-mod021
#COBALT --time 10:00:00
#COBALT --nodecount 1
#COBALT --outputprefix qs-logs/translate-mod021
#COBALT --cwd /home/casper/clouds
#COBALT --attrs=pubnet

source activate clouds

DATA='/projects/CSC249ADCD01/clouds2/mod021km'
OUT='/projects/CSC249ADCD01/clouds2/mod2-tfr'
MODE='mod02_1km'
STRIDES=64 64
SHAPE="182 182"

mpiexec -n 24
    python3 reproduction/pipeline/into_record.py \
		$DATA $OUT $MODE \
		--stride $STRIDES \
 		--shape $SHAPE \
