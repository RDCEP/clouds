#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=gpu2
#SBATCH --account=pi-foster
###SBATCH --time=06:00:00

export KERAS_BACKEND=tensorflow

module purge
module load mpi4py
module load gdal
module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/9.0

source activate clouds

SIDE_LEN=64
NPROCS=8
DATA=/project/foster/clouds/nasa_laads/mod06_l2/order501256920

mpiexec -n $NPROCS python3 hd5_to_TFRecord.py --hdf_dir $DATA