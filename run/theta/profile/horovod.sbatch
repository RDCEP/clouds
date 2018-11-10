#!/bin/sh
#SBATCH --job-name=phor
#SBATCH --output=%x_%A.out
#SBATCH --partition=gpu2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --account=pi-chard
#SBATCH --exclusive


module purge
module load horovod

PFC=/project/foster/clouds
MODEL_PATH=$PFC/output/phor
DATA_DIR=$PFC/data/gee/preprocessed

rm -rf $MODEL_PATH

mpiexec -n $SLURM_NTASKS \
    python reproduction/train.py $MODEL_PATH \
        --data $DATA_DIR/"*".tfrecord \
        --shape 128 128 7 \
        --shuffle_buffer_size 1000 \
        --channel_order channels_last \
        --prefetch 8 \
        --read_threads 16 \
        --n_blocks 4 \
        --block_len 0 \
        --batch_size 104 \
        --dense_ae 256 \
        --base_dim 16 \
        --max_steps 200 \
        --summary_every 25 \
        --save_every 50 \
	--dense_ae 512 \
        --no_augment_rotate \
