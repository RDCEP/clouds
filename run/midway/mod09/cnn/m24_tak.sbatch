#!/bin/sh
#SBATCH --job-name=m9_22
#SBATCH --partition=gpu2
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --account=pi-chard
### -- Longer version of 15b
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
year=2016
month=02    # add 0 less than 10
EXPNAME='Global_ALL_2015_2016'

#BASEFOLDER=/home/rlourenco/rdcep_clouds
BASEFOLDER=/home/tkurihana/clouds
MODEL_PATH=/project/foster/clouds/output/${SLURM_JOB_NAME}_${EXPNAME}
#DATA=/project/foster/clouds/data/2015_05/"*".tfrecord
#DATA=/project2/chard/clouds/data/GEE/clouds_gee_preprocessed_${year}_${month}/"*".tfrecord
DATA=/home/tkurihana/scratch-midway2/data/GEE/clouds_gee_preprocessed_ALL_2015_2016/"*".tfrecord

which mpirun
which python
which nvcc

cd $BASEFOLDER
mpirun -bind-to none -map-by slot  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
python $BASEFOLDER/reproduction/train.py $MODEL_PATH \
    --data $DATA \
    --max_steps 170000 \
    --save_every 5000 \
    --summary_every 250 \
    --shape 128 128 7 \
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
