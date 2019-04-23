#!/bin/bash
#SBATCH --job-name=mod02_prep
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
##SBATCH --partition=broadwl-lc
#SBATCH --partition=broadwl
##SBATCH --ntasks=2
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=42000
#SBATCH --mem-per-cpu=32000

# may be error 1
module load Anaconda3/5.0.0.1
source activate clouds-cpu

# check python
# check difference local or another conda env
which python

#LIB-Path
# may not be necessary
#export PATH=/usr/lib64/nvidia:${PATH}
export LD_LIBRARY_PATH=/usr/lib64/nvidia/libcuda.so.1:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/tkurihana/.conda/envs/clouds-cpu/lib/libgdal.so.20:${LD_LIBRARY_PATH}

# Set directory name parameter
band=28 # 28, 29, 31, or modis06 altitude 
# 28: cirrus band water vapor
# 29: cloud propoerty 
# 31: cloud temperature 


BASEDIR=/home/tkurihana/scratch-midway2/data/MOD02
#OUTPUTBASEDIR=/project2/chard/clouds/data/MOD02   #/home/tkurihana/scratch-midway2/data/MOD02
OUTPUTBASEDIR=/project2/foster/clouds/data/MOD02   #/home/tkurihana/scratch-midway2/data/MOD02
DATA=${BASEDIR}/laads_2000_2018_train
OUT=${OUTPUTBASEDIR}/clouds_laads_preprocessed_2000_2018_band${band}
STRIDES=128
SHAPE=128
PPR=10000 # default PPR=10000

mpiexec -n $SLURM_NTASKS \
        /home/tkurihana/.conda/envs/clouds-cpu/bin/python3 \
        /home/tkurihana/clouds/reproduction/pipeline/into_mod_record.py \
        $DATA/"*".hdf $OUT  \
        --stride $STRIDES \
        --shape $SHAPE \
        --ems_band ${band} \
        --patches_per_record $PPR
