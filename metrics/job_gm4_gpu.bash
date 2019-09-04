#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=loss_rotate_gpu
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=gm4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=01:00:00
#SBATCH --qos=gm4

module load cuda/9.1
module load Anaconda3/5.0.0.1
source activate py3501
# tensorflow 1.12~ gpu version is necessary!

cwd=`pwd`

# directory
logdir=${cwd}/log
figdir=${cwd}/fig
output_modeldir=${cwd}/output_model/${SLURM_JOB_ID}

# params
learning_rate=0.001
num_epoch=2
batch_size=32
copy_size=4
dangle=2
c_lambda=0.1
save_every=1

python train_gpu.py \
  --logdir ${logdir} \
  --figdir ${figdir} \
  --output_modeldir ${output_modeldir} \
  --expname ${SLURM_JOB_ID} \
  --lr ${learning_rate} \
  --num_epoch=${num_epoch} \
  --batch_size=${batch_size} \
  --copy_size=${copy_size} \
  --dangle=${dangle} \
  --c_lambda ${c_lambda} \
  --save_every ${save_every}


cp -r ${cwd}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}* ${output_modeldir}/
