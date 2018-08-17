#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpu2

export KERAS_BACKEND=tensorflow

module purge
module load mpi4py
module load gdal
module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/9.0

source activate clouds

TIF_DATA=/project/foster/clouds/geotiff
TFR_DATA=/project/foster/clouds/geotfr
SIDE_LEN=64
NPROCS=8

mpiexec -n $NPROCS python3 tiff2record.py $TIF_DATA $TFR_DATA --side_len $SIDE_LEN
