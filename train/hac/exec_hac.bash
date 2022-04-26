#!/bin/bash
#
# Add here your scheduler config
# example: 200000GB for 12 days of modis files: 32 hours run time
#          120000GB for  8 days of modis files: 20 hours run time

module load python
source activate tf2-cpu

## basic config
# date of Tfrecord data using for triaing HAC clustering: this date variable is used as an output directory name
date=2003-012-143-190-325_028-159-259-341  # date of clustering training data
# clustering training data directory
TFBASEDIR="/home/tkurihana/scratch-midway2/data/tfrecords"
tf_datadir="${TFBASEDIR}/clouds_terra_laads_multiprocessed_2003_ocean_ncentroids/train_v2_3"
# Trained RI autoencoder model
model_datadir='/project2/foster/clouds/model'
expname='10053302'   # model trained on 10M
layer_name='leaky_re_lu_21'  # NEED layer name in decoder if search_all isn't true
height=128
width=128
channel=6

## Clustering type
clf_key='HAC'  #'DBSCAN' #'HAC'  # HAC

## Cluster numbers
for i in `seq 8 2 61` ; do
  echo $i
  nclusters+=( ${i} )
done
for i in `seq 64 4 129` ; do
  echo $i
  nclusters+=( ${i} )
done
nclusters+=(256)

## COMMON parameter
OUTPUTDIR='/home/tkurihana/scratch-midway2/data/MODIS-CLF'
output_basedir=${OUTPUTDIR}/2003/
cache_datadir="./cache"

## DBSCAN
eps=1.0
leaf_size=30
n_jobs=4
min_samples=5

# memo: nclusters shoud not pu prior to DBSCAN/HAC argparse
python3 prg_hac.py \
        --tf_datadir ${tf_datadir} \
        --output_basedir ${output_basedir} \
        --model_datadir ${model_datadir} \
        --cache_datadir ${cache_datadir} \
        --height ${height} \
        --width ${width} \
        --channel ${channel} \
        --expname ${expname} \
        --clf_key ${clf_key} \
        --resize_flag \
        --layer_name ${layer_name} \
        --nclusters "${nclusters[@]}" \
        --date ${date}  \
        "HAC"
        #"DBSCAN" \
        #--eps ${eps} \
        #--min_samples ${min_samples} \
        #--leaf_size ${leaf_size} \
        #--n_jobs ${n_jobs}
      

#full_tree  # Add if true then compute HAC untill the one cluster
#search_all # Add if true then compute HAC for all activation layer

        #--search_all
        #--full_tree  # attach if only true
        
