#!/bin/bash


##### setting I
scale_patch_size=80  # unit[K-patches] ==>  1,000 patches
# e.g. current maximum is about 1,500. 1,700 will be OOM error
n_cluster=$1
ngroup=$2

echo $n_cluster
echo $ngroup

##### settnig IIu
bands="28_29_31"
homedir="/home/tkurihana/scratch-midway2/anl"
inputdir=${homedir}/output_holdouts/output_clouds_feature_2000_2018_normed/${scale_patch_size}k/group${ngroup}
#inputdir=${homedir}/output_holdouts/${scale_patch_size}k/group${ngroup}
outputdir=${homedir}/output_clustering_2000_2018_m01_band${bands}/sklearn_agglomerative/normed/${scale_patch_size}k/group${ngroup}
#outputdir=${homedir}/output_clustering_2000_2018_m01_band${bands}/sklearn_agglomerative/${scale_patch_size}k/group${ngroup}
model_name='m01_b'${bands}

python prg_sklearn_m02_agglomerative.py \
       --scale_patch_size=${scale_patch_size}  \
       --n_cluster=${n_cluster} \
       --inputdir=${inputdir} \
       --outputdir=${outputdir} \
       --oname=2000-2018_random_aggl \
       --modelname=${model_name}

# 200k memory error 1node-1cpu broadwl OK in Bigmem
#--n_cluster=${SLURM_ARRAY_TASK_ID} \
