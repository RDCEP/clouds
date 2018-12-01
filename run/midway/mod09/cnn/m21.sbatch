#!/bin/sh
#SBATCH --job-name=m9cnn21-dse-henso
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --account=pi-chard
###
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

module load git
module load env/rcc
module load midway2
module load horovod


BASEFOLDER=/home/rlourenco/rdcep_clouds
MODEL_PATH=/project/foster/clouds/output/$SLURM_JOB_NAME
DATA=/project/foster/clouds/data/2015_05/"*".tfrecord

which mpirun
which python
which nvcc

cd $BASEFOLDER
mpirun -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
python $BASEFOLDER/reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --base_dim 8 \
    --max_steps 150000 \
    --save_every 2500 \
    --shape 128 128 7 \
    --autoencoder_adam 0.001 0.9 0.999 \
    --n_blocks 4 \
    --block_len 0 \
    --batchnorm \
    --dense_ae 1024 \
    --read_threads 16 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
    --no_augment_rotate