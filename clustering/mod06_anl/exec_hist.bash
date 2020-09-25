#!/bin/bash

#SBATCH --account=pi-foster
#SBATCH --job-name=hist_mod06
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
##SBATCH --partition=bigmem2
##SBATCH --partition=broadwl
#SBATCH --partition=broadwl-lc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12000 #MB
#SBATCH --time=02:00:00
#SBATCH --array=4,8,16,32,64,128,256,512

module load python/anaconda-2020.02
source activate tf-cpu

# MEMO for partition
# 32 resolution can be done within broadwl but 128 won't: need bigmem
#

echo $SLURM_JOB_ID
# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

tf_datadir="/home/tkurihana/scratch-midway2/data/MOD06"
cluster_datadir='/home/tkurihana/clouds/clustering/large_hac12_2'

## RI
#################expname="67011582"
#expname="4678889"
## NRI
expname='m2_02_global_2000_2018_band28_29_31'

## IMPORTANT parameter
#nclusters=32
#nclusters=2000
nclusters=$SLURM_ARRAY_TASK_ID

## Store key for used slurm job id in clustering
# -- RI --
if [ ${expname} = '4678889' ];then
  declare -A dict=(
      ["4"]="5098535"
      ["8"]="5098536"
      ["16"]="5098537"
      ["32"]="5098538"
      ["64"]="5098539"
      ["128"]="5098540"
      ["256"]="5098541"
      ["512"]="5098542"
      ["1024"]="5098534"
  )
#fi
else
#if [ ${expname} = 'm2_02_global_2000_2018_band28_29_31' ];then
  declare -A dict=(
      ["4"]="5109737"
      ["8"]="5109739"
      ["16"]="5109740"
      ["32"]="5112480"
      ["64"]="5112622"
      ["128"]="5112625"
      ["256"]="5113997"
      ["512"]="5116529"
      ["1024"]="5117581"
  )
fi

echo "count:${#dict[@]}"

#USED_SLURM_JOB_ID="5098538"  #32 for large_hac12_2
USED_SLURM_JOB_ID=`echo "${dict["${SLURM_ARRAY_TASK_ID}"]}"`  #32 for large_hac12_2


## Step size (number of discritization) of bins
nstep=256   # RI

## COMMON parameter
output_basedir='./hist_large_hac12_2'   # large_hac12 ==> copy_size == 12
cache_datadir="./cache/${SLURM_JOB_ID}"
height=128
width=128
channel=6
copy_size=6 # 6 or 180
clf_key='HAC'  # HAC

echo $USED_SLURM_JOB_ID, ${SLURM_ARRAY_TASK_ID}



python3 prg_mod06physic.py \
        --tf_datadir ${tf_datadir} \
        --output_basedir ${output_basedir} \
        --cluster_datadir ${cluster_datadir} \
        --height ${height} \
        --width ${width} \
        --channel ${channel} \
        --expname ${expname} \
        --cexpname ${USED_SLURM_JOB_ID} \
        --copy_size ${copy_size} \
        --nclusters ${nclusters} \
        --nstep ${nstep} \
        --clf_key ${clf_key}
        
