#!/bin/sh
#SBATCH --job-name=m9-22_oceans
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --gres=gpu:1
#SBATCH --account=pi-chard

module load git
module load horovod
###-source activate clouds-gpu

BASEFOLDER=/home/rlourenco/rdcep_clouds
MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA=/project/foster/clouds/data/2015_05"*".tfrecord

### M15 with new data
python $BASEFOLDER/reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --max_steps 100000 \
    --save_every 5000 \
    --summary_every 250 \
    --shape 128 128 7 \
    --autoencoder_adam 0.001 0.9 0.999\
    --base_dim 16 \
    --n_blocks 4 \
    --block_len 0 \
    --batchnorm \
    --read_threads 16 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
    --no_augment_rotate