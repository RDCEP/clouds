#!/bin/sh
#SBATCH --job-name=mod9-1km
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --time 1:00:00

module purge
module load Anaconda3

source activate cc2

OUT=/project/foster/clouds/data/mod9-724x1km/
STRIDES="256 256"
SHAPE="724 724"

mpiexec -n $SLURM_NTASKS \
	 python3 reproduction/pipeline/into_record.py \
		/project/foster/clouds/data/gee/gdrive17AUG2018Afternoon/"*" \
	       	$OUT \
		pptif \
		--stride $STRIDES \
 		--shape $SHAPE \
		--resize 0.5 \
