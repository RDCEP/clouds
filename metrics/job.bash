#!/bin/bash


#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=loss_fn3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=gpu2   #28 cores 64GB memory
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=12:50:00

module load Anaconda3/5.0.0.1
source activate tf-cpu

cwd=`pwd`

# directory
logdir=${cwd}/log
figdir=${cwd}/fig
output_modeldir=${cwd}/output_model/${SLURM_JOB_ID}

# params
learning_rate=0.001
num_epoch=36
batch_size=32
dangle=2
c_lambda=0.1
save_every=5

#memo
# add rotation argument means applying rondom rotation to train data for AE

python train.py \
  --logdir ${logdir} \
  --figdir ${figdir} \
  --output_modeldir ${output_modeldir} \
  --lr ${learning_rate} \
  --num_epoch ${num_epoch} \
  --batch_size ${batch_size} \
  --dangle ${dangle} \
  --c_lambda ${c_lambda} \
  --save_every ${save_every} \
  --rotation

cp -r ${cwd}/loss_fn3_${SLURM_JOB_ID}* ${output_modeldir}/
