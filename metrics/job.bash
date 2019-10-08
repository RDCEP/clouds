#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard

##SBATCH --job-name=loss_rotate
##SBATCH --job-name=loss_reconst
#SBATCH --job-name=loss_full
##SBATCH --job-name=loss_full_restart
##SBATCH --job-name=loss_full_debug
##SBATCH --job-name=loss_log_sum
##SBATCH --job-name=loss_native  # 8G memory is desirable? 6G=OOM
##SBATCH --job-name=loss_lognative  # 8G memory is desirable? 6G=OOM
##SBATCH --job-name=loss_fc

#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=gm4
##SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=4:00:00
##SBATCH --qos=gm4

module load cuda/8.0
module load Anaconda3/5.0.0.1
#source activate tf-cpu
source activate py3501

cwd=`pwd`

# directory
logdir=${cwd}/log
figdir=${cwd}/fig
output_modeldir=${cwd}/output_model/${SLURM_JOB_ID}

# params
learning_rate=0.01 # Adam:0.001 SGD:0.01
num_epoch=50
batch_size=32
copy_size=4
dangle=1
c_lambda=0.05
height=32
width=32
nblocks=5
save_every=1

#python train_debug_restart2.py \
#python train_debug_native.py \
#python train_debug_fc.py \
#python train_debug_log.py \
#python train_debug_lognative.py \
#python train_debug.py \
python train_debug.py \
  --logdir ${logdir} \
  --figdir ${figdir} \
  --output_modeldir ${output_modeldir} \
  --lr ${learning_rate} \
  --num_epoch ${num_epoch} \
  --batch_size ${batch_size} \
  --copy_size=${copy_size} \
  --dangle=${dangle} \
  --c_lambda ${c_lambda} \
  --height ${height} \
  --width ${width} \
  --nblocks ${nblocks} \
  --save_every ${save_every} \
  --expname ${SLURM_JOB_ID} \
  --debug


cp -r ${cwd}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}* ${output_modeldir}/
