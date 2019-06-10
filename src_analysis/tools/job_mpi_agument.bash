#!/bin/bash

#SBATCH --account=pi-foster
#SBATCH --job-name=mpi_agmnt
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=broadwl
##SBATCH --partition=broadwl-lc
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
##SBATCH --mem-per-cpu=42000 #MB

module load Anaconda3/5.0.0.1
source activate tf-cpu

which python

cwd=`pwd`
homedir='/home/tkurihana'
mod02_datadir=${homedir}"/scratch-midway2/data/MOD02/clustering_laads_2000_2018_5"
mod35_datadir=${homedir}"/scratch-midway2/data/MOD35/clustering_laads_2000_2018_5"
#mod02_datadir=${cwd}/"mod02/ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM/2016/001"
#mod35_datadir=${cwd}/"mod35/ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD35_L2/2016/001"
model_dir="/project2/foster/clouds/output/mod02/m2_02_global_2000_2018_band28_29_31"
outputdir=${cwd}'/output_clouds_feature_2000_2018_5'
output_filename='clouds_patches'

#/home/tkurihana/scratch-midway2/anl/prg_augment.py \
mpiexec -n ${SLURM_NTASKS} \
  /home/tkurihana/.conda/envs/tf-cpu/bin/python3 \
  /home/tkurihana/scratch-midway2/anl/prg_augment_2.py \
  --mod02_datadir=$mod02_datadir \
  --mod35_datadir=$mod35_datadir \
  --prefix=MOD35_L2.A \
  --cloud_thres=0.3 \
  --model_dir=${model_dir} \
  --step=100000 \
  --outputdir=${outputdir} \
  --output_filename=${output_filename} 
