#!/bin/bash
#SBATCH --account=pi-foster
#SBATCH --job-name=mod02_prep
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
##SBATCH --partition=broadwl-lc
#SBATCH --partition=broadwl
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000

# may be error 1
#module load Anaconda3/5.0.0.1
#source activate clouds-cpu

SLURM_NTASKS=12

which python

#======================================================
# Set band name parameter
#======================================================
# Thermal band (i.e. band 31) is necessary
#  cloud optical properties are band 6,7,20. 20 should be specified in bands option
bandname='28_29_31' # 28, 29, 31, or modis06 altitude 
bands=( 20 28 29 31) # all thermal/NIR band numbers should be specified. FIX numbers are only 6&7
#bands=(20 29 31)
#++ Band info
# *20: Must include if code does not include band-20
# 28: cirrus band water vapor
# 29: cloud propoerty 
# 31: cloud temperature 


#======================================================
# Set data directory
#======================================================
#BASEDIR=/home/tkurihana/scratch-midway2/data/MOD02
OUTPUTBASEDIR="/home/tkurihana/Research/data/tfrecords"  #/home/tkurihana/scratch-midway2/data/MOD02
#DATA="/home/tkurihana/Research/data2/MOD02/laads_2000_2018_train"
#DATA="/home/tkurihana/Research/data/MOD02/20151201"

DATA="/home/tkurihana/Research/data/MYD02/20080101"
#DATA="/home/tkurihana/Research/data/MYD02/$1"

#OUT=${OUTPUTBASEDIR}/"clouds_laads_preprocessed_2000_2018"
#OUT=${OUTPUTBASEDIR}/"clouds_laads_rawdecoded_2000_2018"
#OUT=${OUTPUTBASEDIR}/"clouds_laads_rawdecoded_20151201"
#OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101"
#OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp"

# special for stride since it's too large
OUTPUTBASEDIR="/home/tkurihana/hdd3/data3"
#OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp_stride"  # smaller stride overlapped each other
#OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_${1}_timestamp_stride"  # smaller stride overlapped each other
#OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp_stride32"  # smaller stride overlapped each other
OUT=${OUTPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp_stride_ocean"  # smaller stride overlapped each other

#mod35_datadir="/home/tkurihana/Research/data2/MOD35/laads_2000_2018_train_all"
mod35_datadir="/home/tkurihana/Research/data/MYD35/20080101"
#mod35_datadir="/home/tkurihana/Research/data/MYD35/${1}"
#stats_datadir="${OUT}/global_mean_std"
stats_datadir=${OUTPUTBASEDIR}/"clouds_laads_rawdecoded_2000_2018/global_mean_std"

#======================================================
# Set parameters
#======================================================
#STRIDES=128
STRIDES=64
#STRIDES=32
SHAPE=128
norm=0 # 0 no / 1 do norm
thres_cloud_frac=0.3
PPR=100000 # default PPR=10000
#PPR=2000 # default PPR=10000
#PPR=491 # default PPR=10000
prefix='MYD35_L2.A' # MOD35_L2.A

# if Aqua
#echo "Aqua Band 6 is contaminated by striping issue: Use band 5 alternatively"




#======================================================
# Set new parameters
#======================================================
thres_ocean_frac=0.999
# append ocean_flag tp argument if you apply ocean flag 

#        /home/tkurihana/clouds/reproduction/pipeline/into_mod_normed_record.py \
#        python3 /home/tkurihana/Research/clouds/reproduction/pipeline/into_mod_logtransform_record.py \
#        python3 /home/tkurihana/Research/clouds/reproduction/pipeline/into_mod_update_record.py \
#        python3 /home/tkurihana/Research/clouds/reproduction/pipeline/into_mod_update_mosaic_record.py \
mpiexec -n $SLURM_NTASKS \
        python3 /home/tkurihana/Research/clouds/reproduction/pipeline/into_mod_update_record_datestamp.py \
        $DATA/"*".hdf $OUT  \
        --stride $STRIDES \
        --shape $SHAPE \
        --ems_band "${bands[@]}" \
        --global_normalization ${norm} \
        --stats_datadir ${stats_datadir} \
        --mod35_datadir ${mod35_datadir} \
        --thres_cloud_frac ${thres_cloud_frac} \
        --prefix ${prefix} \
        --patches_per_record $PPR \
        --thres_ocean_frac ${thres_ocean_frac} \
        --ocean_flag
        
#Takuya's Original script
# /home/tkurihana/clouds/reproduction/pipeline/into_mod_record.py \
