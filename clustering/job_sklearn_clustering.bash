#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=sklearn_aggl
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=broadwl   #28 cores 64GB memory
##SBATCH --partition=broadwl-lc   #28 cores 64GB memory
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=58000 # 1000 = 1GB
#SBATCH --time=00:40:00


#module load spark/2.3.0
#module load Anaconda3/5.0.0.1
#source activate tf-cpu

which python

cwd=`pwd`

##### setting I
scale_patch_size=18  # unit[K-patches] ==>  1,000 patches
# e.g. current maximum is about 1,500. 1,700 will be OOM error
n_cluster=12

##### settnig II
bands="28_29_31"
inputdir=${cwd}/outputs/output_clouds_feature_20151116-20151215
outputdir=${cwd}/output_clustering_2000_2018_m02_band${bands}/sklearn_agglomerative/${scale_patch_size}k
model_name='m02_b'${bands}

python prg_sklearn_m02_agglomerative.py \
       --scale_patch_size=${scale_patch_size}  \
       --n_cluster=${n_cluster} \
       --inputdir=${inputdir} \
       --outputdir=${outputdir} \
       --oname=2000-2018_random_aggl \
       --modelname=${model_name}
