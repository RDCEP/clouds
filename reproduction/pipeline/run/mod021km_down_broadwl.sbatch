#!/bin/bash
#SBATCH --job-name=M21Jun
###SBATCH --output=MOD021km.out
###SBATCH --error=MOD021km.err
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err
#SBATCH --partition=broadwl
###SBATCH --time=06:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000

#SBATCH --account=pi-chard
###SBATCH --time=06:00:00

module purge
module load mpi4py/3.0.1a0_py3

module unload python
module unload Anaconda2
module unload Anaconda3

### Routine Using rlourenco Miniconda env
source activate python3env

mpiexec -n 1  python3 mod021km_download.py

