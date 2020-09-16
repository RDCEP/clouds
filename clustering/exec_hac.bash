#!/bin/bash

#SBATCH --account=pi-foster
##SBATCH --job-name=big_rot_hac # large_hac2
#SBATCH --job-name=big_sel_hac # large_hac3(2-10.tf),4(3-0.tf)
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=bigmem2
##SBATCH --partition=broadwl
##SBATCH --partition=broadwl-lc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=32000 #MB
##SBATCH --mem-per-cpu=48000 #MB
##SBATCH --mem-per-cpu=50000 #MB
##SBATCH --mem-per-cpu=80000 #MB
#SBATCH --mem-per-cpu=140000 #MB
#SBATCH --time=01:00:00
##SBATCH --array=2,4,8,16,32,64,128,256
##SBATCH --array=2,4,8,16,32,64,128
##SBATCH --array=32,64,128
##SBATCH --array=300,600,1200,1800
##SBATCH --array=4,8,16,64,128
##SBATCH --array=1200,1800

module load python/anaconda-2020.02
source activate tf-cpu

# MEMO for partition
# 32 resolution can be done within broadwl but 128 won't: need bigmem
#

echo $SLURM_JOB_ID
# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

## RI
#tf_datadir="/project2/foster/clouds/data/clouds_laads_multiprocessed_2000_2018_band28_29_31_circle_2"
#model_datadir='/home/tkurihana/rotate_invariant/stepbystep/transform/output_model'
#expname="67011582"
#height=32
#width=32


## NRI
tf_datadir="/project2/foster/clouds/data/clouds_laads_rawdecoded_2000_2018"  
model_datadir='/home/tkurihana/rotate_invariant/stepbystep/transform/output_model'
expname='m2_02_global_2000_2018_band28_29_31'
height=128
width=128

## IMPORTANT parameter
nclusters=1800
#nclusters=900  # hardcode npatches  == 300 x number of copy size 1/2 == 3
#nclusters=300  # hardcode npatches  == 300
#nclusters=674    # Max 674 for 2-10 patch under min-std 0.08, min-mean 0.1 max-mean 0.9 
#nclusters=$SLURM_ARRAY_TASK_ID
alpha=100
#full_tree # Add if true then compute HAC untill the parameter

## COMMON parameter
output_basedir='./large_hac4'
cache_datadir="./cache/${SLURM_JOB_ID}"
channel=6
copy_size=12 # or 180
clf_key='HAC'  # HAC

python3 prg_hac.py \
        --tf_datadir ${tf_datadir} \
        --output_basedir ${output_basedir} \
        --model_datadir ${model_datadir} \
        --cache_datadir ${cache_datadir} \
        --height ${height} \
        --width ${width} \
        --channel ${channel} \
        --expname ${expname} \
        --cexpname ${SLURM_JOB_ID} \
        --copy_size ${copy_size} \
        --nclusters ${nclusters} \
        --alpha ${alpha} \
        --clf_key ${clf_key}
        #--full_tree  # attach if only true
        
