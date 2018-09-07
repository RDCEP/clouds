#!/bin/sh
#SBATCH --job-name=mod09
#SBATCH --output=%x_%A.out
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=gpu2
#SBATCH --ntasks=28
#SBATCH --mem-per-cpu=10000

module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
#module load mpi4py
#module load gdal


source activate clouds

DAT=/project/foster/clouds/data/gee/gdrive17AUG2018Afternoon/
OUT=/project/foster/clouds/data/gee/records

mpiexec -n $SLURM_NTASKS python3 reproduction/pipeline/into_record.py \
    --tif_dir $DAT \
    --out_dir $OUT 
