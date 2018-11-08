#!/bin/sh
#SBATCH --job-name=m021km
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --time 3:00:00
#SBATCH --account=pi-chard

module purge
module load Anaconda3
module load mpi4py
source activate cc2

DATA=/scratch/midway2/rlourenco/mod021km
OUT=$SCRATCH/m2tfr
MODE=mod02_1km
STRIDES="128 128"
SHAPE="182 182"

mpiexec -n $SLURM_NTASKS \
    python3 reproduction/pipeline/into_record.py \
        $DATA/"*".hdf $OUT $MODE \
        --stride $STRIDES \
        --shape $SHAPE \
