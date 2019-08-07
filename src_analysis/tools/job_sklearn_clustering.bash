#!/bin/bash

#SBATCH --account=pi-foster       # default pi-foster
#SBATCH --job-name=sklearn_aggl   # job name
#SBATCH --output=%x_%A_%a.out     # log file name
#SBATCH --error=%x_%A_%a.err      # error file name
#SBATCH --partition=broadwl       # 1 node = 28 cores 64GB memory
#SBATCH --nodes=1                 # number of node
#SBATCH --cpus-per-task=1         # cpu per task
#SBATCH --mem-per-cpu=58000       # 1024 = 1GB Max 58000 ~ 80-90K patches
#SBATCH --time=03:00:00           # wall time. MUST set!

# + comments for ruby
#   1. Interactive Job
#   If you use interactive job, you will type the above SBATCH options
#   to initiate interactive mode e.g. sinteractive --account=pi-foster ... --time=03:00:00
#   There is no big difference between sbatch and sinteractive.
#   Sometime when many people submit their jobs, you hardly connect to interactive mode immediately
#
#
#   2. Resource management
#   Please make sure to manage how much SUs did you used (this job will up to several hundres SUs)
#   Follwoing commands will helpful (type on terminal)  
#     rcchelp balance      --> how much group keeps/comsumed SUs
#     rcchelp usage -byjob --> how much each job consumed SUs
#



# Fill out following settings
##### setting I
scale_patch_size=80  # unit[K-patches] ==>  1,000 patches
n_cluster=20  # you can choose any cluster number 
ngroup=0      # group number if you need specify. 4 groups from group 0-3

# check  
echo $n_cluster
echo $ngroup

##### settnig II
bands="28_29_31"
homedir="/home/tkurihana/scratch-midway2/anl"
inputdir=${homedir}/output_holdouts/${scale_patch_size}k/group${ngroup}
outputdir=${homedir}/output_clustering_2000_2018_m01_band${bands}/sklearn_agglomerative/${scale_patch_size}k/group${ngroup}
model_name='m01_b'${bands}

# + comments for ruby
#
#  Points that you should change are 
#    1. homedir (actually not necessary. Just for avoiding duplication of same path)
#       e.g. /project2/foster/clouds/analysis
#
#    2. inputdir (MUST)
#       e.g. ${homedir}/output_clouds_feature_2000_2018_validfiles
#         or ${homedir}/output_clouds_feature_2000_2018_validfiles/group0_mix/ here you may copy/link files to this directory
#             * mix means mix of patch files with your data
#
#    3. outputdir (MUST)
#       e.g. /home/ruby/scratch-midway2/output_mix_clustering
#


python prg_sklearn_m02_agglomerative.py \
       --scale_patch_size=${scale_patch_size}  \
       --n_cluster=${n_cluster} \
       --inputdir=${inputdir} \
       --outputdir=${outputdir} \
       --oname=2000-2018_random_aggl \
       --modelname=${model_name}

# 200k memory error 1node-1cpu broadwl OK in Bigmem
#--n_cluster=${SLURM_ARRAY_TASK_ID} \
