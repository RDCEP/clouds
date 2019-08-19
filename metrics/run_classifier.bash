#!/bin/bash


#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=loss_cnn_norotate
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=gpu2   #28 cores 64GB memory
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=01:00:00

module load Anaconda3/5.0.0.1
source activate tf-cpu

cwd=`pwd`

# directory
logdir=${cwd}/log
output_modeldir=${cwd}/output_model/${SLURM_JOB_ID}
#model_dir=${cwd}/output_model/original2
model_dir=${cwd}/output_model/61881099
# 61881099 lr0.001-lambda0.1

# params
learning_rate=0.001
num_epoch=31
batch_size=32
save_every=5
step=35
depth=10

# memo
# when won't apply random rotation, comment out --rotation argument

# set rotation oprion when apply rotation operation to training data
python classifier.py \
  --logdir ${logdir} \
  --model_dir ${model_dir} \
  --output_modeldir ${output_modeldir} \
  --lr ${learning_rate} \
  --num_epoch ${num_epoch} \
  --batch_size ${batch_size} \
  --save_every ${save_every} \
  --shape 7 7 10 \
  --depth ${depth} \
  --step ${step}
  #--rotation

cp -r ${cwd}/loss_cnn_${SLURM_JOB_ID}* ${output_modeldir}/
