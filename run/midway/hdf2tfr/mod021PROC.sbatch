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

#======================================================
# Set parallel config
#======================================================
SLURM_NTASKS=6


#======================================================
# Set data directory
#======================================================
#INPUTBASEDIR="/home/tkurihana/Research/data/tfrecords"  #/home/tkurihana/scratch-midway2/data/MOD02
INPUTBASEDIR="/home/tkurihana/hdd3/data3"  #/home/tkurihana/scratch-midway2/data/MOD02
#DATA=${INPUTBASEDIR}/"clouds_laads_rawdecoded_2000_2018"
#DATA=${INPUTBASEDIR}/"clouds_laads_rawdecoded_20151201"
#DATA=${INPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20160101"
#DATA=${INPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101"
#DATA=${INPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp"
#DATA=${INPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp_stride"
#DATA=${INPUTBASEDIR}/"clouds_MYD_laads_rawdecoded_20080101_timestamp_stride32"
#DATA=${INPUTBASEDIR}/${1}
#DATA=${INPUTBASEDIR}/"clouds_laads_rawdecoded_2000_2018/MoreThan30files"
DATA="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20080101_timestamp_stride_ocean"

#stats_datadir="${DATA}/global_mean_std"
#stats_datadir="${INPUTBASEDIR}/clouds_laads_rawdecoded_2000_2018/global_mean_std"
#stats_datadir="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20160101/global_mean_std"
#stats_datadir="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20080101/global_mean_std"
#stats_datadir="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20080101_timestamp_stride/global_mean_std"
#stats_datadir="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20080101_timestamp_stride32/global_mean_std"
stats_datadir="${INPUTBASEDIR}/clouds_MYD_laads_rawdecoded_20080101_timestamp_stride_ocean/global_mean_std"

#OUTPUTBASEDIR="/home/tkurihana/Research/data/tfrecords"
OUTPUTBASEDIR="/home/tkurihana/hdd3/data3" 
#OUT="${OUTPUTBASEDIR}/clouds_laads_multiprocessed_2000_2018" # scaling + circle-mask
#OUT="${OUTPUTBASEDIR}/clouds_laads_multiprocessed_2000_2018_band6_7" # scaling + circle-mask
#OUT="${OUTPUTBASEDIR}/clouds_laads_multiprocessed_2000_2018_band6" # scaling + circle-mask
#OUT="${OUTPUTBASEDIR}/clouds_laads_multiprocessed_2000_2018_2" # scaling + circle-mask for TI-preprocessing
#OUT="${OUTPUTBASEDIR}/clouds_laads_multiprocessed_20151201" # scaling + circle-mask
#OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20080101" # scaling + circle-mask
#OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20080101_timestamp" # scaling + circle-mask for each timestamp
#OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20080101_timestamp_stride" # scaling + circle-mask for each timestamp
#OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20080101_timestamp_stride32" # scaling + circle-mask for each timestamp
#OUT="${OUTPUTBASEDIR}/${2}"     # scaling + circle-mask for each timestamp
#OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20160101" # scaling + circle-mask
OUT="${OUTPUTBASEDIR}/clouds_MYD_laads_multiprocessed_20080101_timestamp_stride_ocean" # scaling + circle-mask for each timestamp

#======================================================
# Set parameters
#======================================================
channels=6  # all bands in original raw cloud data regardless of bands for scaled output
nsigma=2
#sbands=( 0 1 ) # index of band;
sbands=( 0 ) # index of band;
#PPR=2000 # default PPR=10000
PPR=100000 # default PPR=10000
#PPR=491 # default PPR=10000

echo $OUT ${DATA}


mpiexec -n $SLURM_NTASKS \
        python3 /home/tkurihana/Research/clouds/reproduction/pipeline/int_mod_scale_record.py \
        $DATA/"*".tfrecord $OUT  \
        --channels ${channels} \
        --nsigma ${nsigma} \
        --stats_datadir ${stats_datadir} \
        --sbands "${sbands[@]}" \
        --patches_per_record $PPR
