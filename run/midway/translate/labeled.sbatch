#!/bin/sh
#SBATCH --job-name=tr-labeled
#SBATCH --output=%x_%A.out
#SBATCH --partition=bigmem2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time 1:00:00

module purge
module load Anaconda3

source activate cc2

CLOS=/project/foster/clouds/data/closed_cell-128x500m/
OPEN=/project/foster/clouds/data/open_cell-128x500m/

DATA=/project/foster/clouds/data/gee/closed_open_cell/2017_01_MOD09GA/



STRIDES="256 256"
SHAPE="724 724"

mpiexec -n $SLURM_NTASKS \
	 python3 reproduction/pipeline/into_record.py \
                $DATA"*open-cell.tif"\
	        $OPEN \
		pptif \
		--stride $STRIDES \
 		--shape $SHAPE \

#mpiexec -n $SLURM_NTASKS \
#	python3 reproduction/pipeline/into_record.py \
#		$DATA"*closed-cell.tif" \
#		$CLOS \
#		pptif \
#		--stride $STRIDES \
#		--shape $SHAPE \
