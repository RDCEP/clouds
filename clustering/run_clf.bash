#!/bin/bash

#SBATCH --account=pi-foster
#SBATCH --job-name=mpl_clf
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=broadwl
##SBATCH --partition=broadwl-lc
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=12000 #MB
##SBATCH --mem-per-cpu=32000 #MB
#SBATCH --mem-per-cpu=42000 #MB
#SBATCH --time=10:00:00

module load python/anaconda-2020.02
source activate tf-cpu

echo $SLURM_JOB_ID
echo $SLURM_NTASKS

## RI
#tf_datadir="/project2/foster/clouds/data/clouds_laads_multiprocessed_2000_2018_band28_29_31_circle_2"
#model_datadir='/home/tkurihana/rotate_invariant/stepbystep/transform/output_model'
#expname="67011582"
#height=32
#width=32


## NRI
tf_datadir="/project2/foster/clouds/data/clouds_laads_rawdecoded_2000_2018"
model_datadir='/home/tkurihana/rotate_invariant/stepbystep/transform/output_model'
expname='m2_02_global_2000_2018_band28_29_31'
height=128
width=128

## COMMON parameter
#output_basedir='./cross_validation_index6_nri'
output_basedir='./cross_validation_index6_ri'
#output_basedir='./cross_validation_texture2'
channel=6
copy_size=360 # or 180
cv=3 # or 4 ,5- return almost 1
clf_key='SVM'  # SVM, MLP, RF, ADABOOST

mpiexec -n $SLURM_NTASKS \
python3 classifier.py \
        --tf_datadir ${tf_datadir} \
        --output_basedir ${output_basedir} \
        --model_datadir ${model_datadir} \
        --height ${height} \
        --width ${width} \
        --channel ${channel} \
        --expname ${expname} \
        --cexpname ${SLURM_JOB_ID} \
        --copy_size ${copy_size} \
        --cv ${cv} \
        --clf_key ${clf_key} \
        --nproc ${SLURM_NTASKS}
        
