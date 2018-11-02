#!/bin/sh
#SBATCH --job-name=m021km
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --time 3:00:00
#SBATCH --account=pi-chard

module purge
module load Anaconda3

source activate

DATA=$SCRATCH/mod021km
OUT=$SCRATCH/mod2-tfr
MODE=mod02_1km
STRIDES="64 64"
SHAPE="182 182"

mpiexec -n $SLURM_NTASKS \
	 python3 reproduction/pipeline/into_record.py \
		$DATA $OUT $MODE \
		--stride $STRIDES \
 		--shape $SHAPE \
