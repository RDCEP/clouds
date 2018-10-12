#!/bin/sh
#SBATCH --job-name=cluster
#SBATCH --output=%x_%A.out
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000

module purge
module load Anaconda3

source activate cc2

PFC=/project/foster/clouds
DATA=$PFC/data/gee/gdrive17AUG2018Afternoon/*.tfrecord
ENCODER=$PFC/output/mod09cnn17
OUT=$PFC/cluster/c1

python3 reproduction/clustermodels.py \
	$OUT $ENCODER \
	--batch_size 1000 \
	--data $DATA \
	--save_every 50
