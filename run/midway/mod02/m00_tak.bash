#!/bin/sh
#SBATCH --job-name=m2_00
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --account=pi-chard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
####SBATCH --mem-per-cpu=32000
#SBATCH --mem-per-cpu=16000
####SBATCH --gres=gpu:2
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00

module load git
module load env/rcc
module load midway2
module load horovod

# set parameter
band=28  # tmp/alt band for cloud top feature extraction
EXPNAME='global_2000_2018_band'${band}

BASEFOLDER=/home/tkurihana/clouds
MODEL_PATH=/project2/foster/clouds/output/mod02/${SLURM_JOB_NAME}_${EXPNAME}
#DATA=/project2/chard/clouds/data/MOD02/clouds_laads_preprocessed_2000_2018_band${band}/"*".tfrecord
DATA=/project2/foster/clouds/data/MOD02/clouds_laads_preprocessed_2000_2018_band${band}/"*".tfrecord

which mpirun
which python
which nvcc

cd $BASEFOLDER
mpirun -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
python $BASEFOLDER/reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --max_steps 110000 \
    --save_every 2500 \
    --summary_every 250 \
    --shape 128 128 4 \
    --autoencoder_adam 0.001 0.9 0.999\
    --base_dim 16 \
    --n_blocks 4 \
    --block_len 0 \
    --batchnorm \
    --read_threads 64 \
    --shuffle_buffer_size 1000 \
    --image_loss_weights 1 1 1 1 \
    --no_augment_rotate

#
#    --max_steps 200000 \
