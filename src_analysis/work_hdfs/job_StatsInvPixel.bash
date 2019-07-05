#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=stats_sds
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=broadwl   #28 cores 64GB memory
##SBATCH --partition=broadwl-lc   #28 cores 64GB memory
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
##SBATCH --exclusive


module load Anaconda3/5.0.0.1
source activate tf-cpu # maybe clouds in your case

which python

# params
# Set parameter/directory below
mod02_datadir=`pwd`
mod35_datadir='/home/tkurihana/scratch-midway2/data/MOD35/laads_2000_2018_train'
outputdir=`pwd`/output
outputname='stats'


python prg_StatsInvPixel.py \
  --mod02_datadir=${mod02_datadir} \
  --mod35_datadir=${mod35_datadir} \
  --outputdir=${outputdir} \
  --outputname=${outputname} 
