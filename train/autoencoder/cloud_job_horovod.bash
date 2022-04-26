#!/bin/bash
#
# Specify here your HPC job scheduler option
# 

## ++ MEMO ++
# an example of submission
# qsub -t 01:00:00 -n 2 -A  Clouds --jobname=1M-16gpus  cloud_job_horovod.bash 16
# ngpus = $1

echo "Setting up virtial env:"
source "your conda env"
cwd=`pwd`

# directory
logdir=${cwd}/log
figdir=${cwd}/fig
output_modeldir=${cwd}/output_model/${COBALT_JOBID}

rexpname="10053121"  # JOB scheduler ID assigned to the previous job e.g. --expname ${COBALT_JOBID}
                     # this is necessary if you load previous trained weights and retrain from the weights.
tf_datadir="/lus/eagle/projects/Clouds"
input_datadir=( 
  # list of directory
  "/home/data/A"
  "/home/data/B"
)


retrain_datadir="${cwd}/output_model/${rexpname}"
#retrain_datadir="${cpd}/output_model/${rexpname}"

# params
learning_rate=0.01 # Adam:0.001 SGD:0.01
num_epoch=100   # 96min for 25 epochs on one-V100
batch_size=128  # fit in 16 V100 gpus  
copy_size=1
echo " ### NUMBER OF GPU =  $1  BATCH SIZE  $batch_size ###"

#### ===================================== 
#         Model parameters
#### =====================================
# lambda(inv, res) = (32,80) is the best combination reported Kurihana et al. 2021
## First coef: lambda_inv
f_lambda=32.0 
## Second coef: lambda_res
c_lambda=80.0
## Third coef
s_lambda=0.0

#### ===================================== 
#         Model parameters
#### =====================================
## For image
# ------------------------- Resolution ----------------------
height=32      # (resized) height of input image 128pixel --> Xpixel
width=32       # (resized) width  of input image
# ---------------------------------------------------------------
channel=6      # clouds
npatches=10000000  # size of entire patches in a list of tfrecords 

## For model arch.
nblocks=5  # 32x32 for 4x4 x256
base_dim=5  # 32x32-->2,2,512 for nblocks 5 1x1x1024 for nblock 5 #### <--- This is default
nstack_layer=3   # number of conv. layers/block

## Other
save_every=10   # save every n epochs
degree=5        # rotate every n degree

# you may want to change option for mpirun based on your HPC system
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $1 -npernode 8 --hostfile $COBALT_NODEFILE  \
  python $cwd/train_cloud_tf2_custom.py \
  --logdir ${logdir} \
  --figdir ${figdir} \
  --input_datadir ${input_datadir} \
  --output_modeldir ${output_modeldir} \
  --lr ${learning_rate} \
  --num_epoch ${num_epoch} \
  --batch_size ${batch_size} \
  --npatches ${npatches} \
  --copy_size=${copy_size} \
  --f_lambda ${f_lambda} \
  --c_lambda ${c_lambda} \
  --s_lambda ${s_lambda} \
  --degree ${degree} \
  --height ${height} \
  --width ${width} \
  --channel ${channel} \
  --nblocks ${nblocks} \
  --base_dim ${base_dim} \
  --nstack_layer ${nstack_layer} \
  --save_every ${save_every} \
  --retrain_datadir ${retrain_datadir} \
  --expname ${COBALT_JOBID}  \
  --retrain # comment of --retrain if you retrain from existed weights
  #--resnet # *(this is optional) if you want to try training residual network, add this option