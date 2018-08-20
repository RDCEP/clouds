#!/bin/bash
#SBATCH --job-name=HDFtoTFR
###SBATCH --output=example_single_node_sbatch.out
###SBATCH --error=example_single_node_sbatch.err
#SBATCH --partition=broadwl
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000

#SBATCH --account=pi-foster
###SBATCH --time=06:00:00

export KERAS_BACKEND=tensorflow

module purge
module load mpi4py/3.0.1a0_py3

module unload python
module unload Anaconda2
module load Anaconda3/5.0.0.1

module load graphviz
module load cuda/9.0

source activate clouds_casper

SIDE_LEN=64
NPROCS=8
DATA=/project/foster/clouds/nasa_laads/mod06_l2/2017_01_01

mpiexec -n $NPROCS python3 hd5_to_TFRecord.py --with_meta --bytes --hdf_dir $DATA