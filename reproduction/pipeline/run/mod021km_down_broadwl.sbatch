#!/bin/bash
#SBATCH --job-name=MOD021km
#SBATCH --output=MOD021km.out
#SBATCH --error=MOD021km.err
#SBATCH --partition=broadwl
###SBATCH --time=06:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000

#SBATCH --account=pi-foster
###SBATCH --time=06:00:00

module purge
module load mpi4py/3.0.1a0_py3

module unload python
module unload Anaconda2
module load Anaconda3/5.0.0.1

source activate python3env


###mpiexec -n $NPROCS python3 hd5_to_TFRecord.py --with_meta --bytes --hdf_dir $DATA

mpiexec -n $NPROCS python3 download_mpi.py

#### - srun --ntasks 31 -p broadwl mpiexec -n 31 python3 download_mpi.py
