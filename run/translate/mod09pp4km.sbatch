#!/bin/sh
#SBATCH --job-name=4km_pxl
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000

module purge
module load Anaconda3

source activate cc2

DAT=/project/foster/clouds/data/gee/gdrive17AUG2018Afternoon/
OUT=/project/foster/clouds/data/gee/4km_pxl
STRIDES="128 128"
SHAPE="128 128"
PPR=10000

mpiexec -n $SLURM_NTASKS python3 reproduction/pipeline/into_record.py $DAT pptif --out_dir $OUT --stride $STRIDES --shape $SHAPE --patches_per_record $PPR --resize 0.25
