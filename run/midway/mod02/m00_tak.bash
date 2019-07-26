#!/bin/sh
#SBATCH --account=pi-foster
#SBATCH --job-name=m2_04
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
####SBATCH --mem-per-cpu=32000
#SBATCH --mem-per-cpu=16000
##SBATCH --gres=gpu:2
#SBATCH --gres=gpu:4
###SBATCH --time=36:00:00
#SBATCH --time=16:00:00

module load git
module load env/rcc
module load midway2
module load horovod

# model
# ++ DO NOT forget change "shape" 
#00 4bands
#01 5bands
#02 6bands
#03 3bands
#04 6bands + 7layer


# set parameter
band='28_29_31'  # tmp/alt band for cloud top feature extraction
EXPNAME='global_2000_2018_band'${band}
#EXPNAME='test_band'${band}

BASEFOLDER=/home/tkurihana/clouds
MODEL_PATH=/project2/foster/clouds/output/mod02/${SLURM_JOB_NAME}_${EXPNAME}
#DATA=/project2/chard/clouds/data/MOD02/clouds_laads_preprocessed_2000_2018_band${band}/"*".tfrecord
DATA=/project2/foster/clouds/data/MOD02/clouds_laads_preprocessed_2000_2018_band${band}/"*".tfrecord
#DATA=/home/tkurihana/scratch-midway2/data/MOD02/clouds_laads_preprocessed_2000_2018_band${band}/"*".tfrecord
#DATA=/home/tkurihana/scratch-midway2/data/MOD02/test_num_variables/"*".tfrecord

which mpirun
which python
which nvcc

cd $BASEFOLDER
mpirun -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
python $BASEFOLDER/reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --max_steps 105000 \
    --save_every 2500 \
    --summary_every 250 \
    --shape 128 128 6 \
    --autoencoder_adam 0.001 0.9 0.999\
    --base_dim 16 \
    --n_blocks 7 \
    --block_len 0 \
    --batchnorm \
    --read_threads 64 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
    --no_augment_rotate

### General settings
#    --max_steps 200000 \
#    --max_steps 110000 \
#    --save_every 2500 \
#    --summary_every 250 \
#    --n_blocks 4 \
#    --shuffle_buffer_size 1000 \
