#!/bin/sh
#SBATCH --job-name=mod09pp
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=broadwl
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000

source activate clouds-cpu

which python

BASEFOLDER=/home/rlourenco/rdcep_clouds
### On DATA, always include the extension, or asterisk for all files
DATA=$SCRATCH/clouds_high_enso
OUT=$SCRATCH/clouds_high_enso_preprocessed
MODE=mod09_tif
STRIDES="128 128"
SHAPE="128 128"
PPR=10000


mpiexec -n $SLURM_NTASKS \
    /home/rlourenco/miniconda3/envs/clouds-cpu/bin/python3 /home/rlourenco/rdcep_clouds/reproduction/pipeline/into_record.py \
        $DATA/"*".tif $OUT $MODE \
        --stride $STRIDES \
        --shape $SHAPE \
        --patches_per_record $PPR
