#!/bin/bash
#SBATCH --account=pi-foster
#SBATCH --partition=broadwl
##SBATCH --nodes=60
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=29000 # 1000 = 1GB

echo "Number of procs : ${SLURM_NTASKS} "

module load Anaconda3/5.0.0.1
source activate tf-cpu

mod02_datadir="/home/tkurihana/scratch-midway2/data/MOD02/regression_laads_2000_2018_1"
mod35_datadir="/home/tkurihana/scratch-midway2/data/MOD35/regression_laads_2000_2018_1"
#tfrecord_datadir="/project2/chard/clouds/data/MOD02/MOD02/clouds_laads_preprocessed_2000_2018_band28_29_31"
tfrecord_datadir="./"
outputdir=${tfrecord_datadir}"/global_mean_std"
outputfname="m2_02_band28_29_31"  # model name + band

mpiexec -n $SLURM_NTASKS  \
  python prg_mod_normed_globalMeanStd.py \
  --mod02_datadir=${mod02_datadir} \
  --mod35_datadir=${mod35_datadir} \
  --tfrecord_datadir=${tfrecord_datadir} \
  --outputdir=${outputdir} \
  --outputfname=${outputfname} \
  --operate_single=0

# memo
# 28G ==> ~10,000 patches?
# 28G is not affordable for 2 files (20,000 patches)

# original test script
#  python test_mpi_globalMeanStd.py \
