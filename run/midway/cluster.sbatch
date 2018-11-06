#!/bin/sh
#SBATCH --job-name=cluster
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000
#SBATCH --account=pi-chard
##SBATCH --time=01:00:00

module purge
module load Anaconda3

source activate cc4

PFC=/project/foster/clouds
DATA=$PFC/data/gee/preprocessed/'*'.tfrecord
ENCODER=$PFC/output/mod09cnn17
OUT=$PFC/cluster/c3


python reproduction/clustermodels.py \
	$OUT $ENCODER sparse_dict \
	--batch_size 3 \
	--data $DATA \
	--shape 128 128 7 \
	--no_augment_rotate \
	--save_every 50
