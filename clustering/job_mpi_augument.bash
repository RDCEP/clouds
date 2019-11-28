#!/bin/bash

#SBATCH --account=pi-foster
#SBATCH --job-name=mpi_agmnt
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=broadwl
##SBATCH --partition=broadwl-lc
#SBATCH --ntasks=8
#SBATCH --nodes=4
##SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=28000 #MB
##SBATCH --mem-per-cpu=42000 #MB
#SBATCH --time=02:00:00

#module load Anaconda3/5.0.0.1
#source activate tf-cpu

SLURM_NTASKS=5

which python


cwd=`pwd`
homedir="/home/tkurihana/Research"
#mod02_datadir=${homedir}"/data/MOD02/2010-2012-2015-JJA"
mod02_datadir=${homedir}"/data/MOD02/2010-2012-2015-JJA_tmp"
mod35_datadir=${homedir}"/data/MOD35/2010-2012-2015-JJA"
model_dir=${homedir}"/models/m2_02_global_2000_2018_band28_29_31_normed"
#model_dir=${homedir}"/models/rt0_63881259_global_2000_2018_band28_29_31"
outputdir=${cwd}"/outputs_normed/output_clouds_feature_2010-2012-2015-JJA/"
#outputdir=${cwd}"/outputs_rotate-invariant/output_clouds_feature_20151116-20151215/"
output_filename='clouds_patches'

# normalization
normed=1 # 1: normalization / 0: NO-normalization
base_stats_datadir="/home/tkurihana/Research/data/tfrecords/clouds_laads_preprocessed_2000_2018_m2normed_2010-2012-2015-JJA"
stats_datadir=${base_stats_datadir}/global_mean_std

# number of epoch/step
step=100000

#/home/tkurihana/scratch-midway2/anl/prg_augment.py \
time mpiexec -n ${SLURM_NTASKS} \
  python prg_augment.py \
  --mod02_datadir=$mod02_datadir \
  --mod35_datadir=$mod35_datadir \
  --prefix=MOD35_L2.A \
  --cloud_thres=0.3 \
  --model_dir=${model_dir} \
  --step=${step} \
  --outputdir=${outputdir} \
  --output_filename=${output_filename} \
  --normed=${normed} \
  --stats_datadir=${stats_datadir}

#  --step=100000 \
