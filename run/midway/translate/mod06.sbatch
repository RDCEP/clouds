#!/bin/sh
#SBATCH --job-name=into_records
#SBATCH --output=%x_%A.out
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=gpu2
#SBATCH --ntasks=28


module purge
module load Anaconda3/5.0.0.1
module load cuda/9.0
#module load mpi4py
#module load gdal


source activate clouds

DATA_DIRS=/project/foster/clouds/data/nasa/mod06_l2/2017_*
OUT=/project/foster/clouds/data/nasa/mod06_l2/small_recs/
FIELDS='Cloud_Optical_Thickness Cloud_Water_Path Cloud_Effective_Radius'

for DIR in $DATA_DIRS
do
    mpiexec -n $SLURM_NTASKS python3 reproduction/pipeline/into_record.py \
        --fields $FIELDS \
        --hdf_dir $DIR \
        --out_dir $OUT 
done
