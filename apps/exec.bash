#!/bin/bash

## Config
# argument passed from exec_array.bash
year=$1 
date=$2
nclusters=$3
clusterstamp=$4
cwd=$5
ndays=$6
instrument=$7 #'terra' or 'aqua'

# Parsl config on HPC
cores_per_node=2    # On theta at ANL 64 cores, 192GiB memory. Choose optimal numbers based on your HPC systems
cores_per_worker=1  # always 1
nodes_per_block=128 # number of computer nodes on your HPC systems
max_workers=2       # number of workers running at the same time

# following may vary on your system
queue='default'
walltime='00:43:00'

## Directories
# mean and standard deviation data: this is needed to normalize patch data
app_homedir='/gpfs/mira-home/kurihana/clouds_project/modis-climate'
tf_datadir="${app_homedir}/common/global_mean_std/${instrument}"

# MODIS data storge
if [ ${instrument} == 'terra' ] ; then
    hdf_mod02_datadir="/eagle/Clouds/C6.1/L1/MOD021KM/${year}"
    hdf_mod03_datadir="/eagle/Clouds/C6.1/L1/MOD03"
    hdf_mod06_datadir="/eagle/Clouds/C6.1/L2/MOD06_L2"
elif [ ${instrument} == 'aqua' ] ; then
    hdf_mod02_datadir="/eagle/Clouds/C6.1/L1/MYD021KM/${year}"
    hdf_mod03_datadir="/eagle/Clouds/C6.1/L1/MYD03"
    hdf_mod06_datadir="/eagle/Clouds/C6.1/L2/MYD06_L2"
fi

## Outputdir
outputbasedir="/eagle/Clouds/C6.1-tak/outputs/modis-20years/${clusterstamp}/${year}"

## Centroids
centroids_filename="${app_homedir}/apps/hac/${clusterstamp}/hac_ncluster${nclusters}-centroids-10021600.npy"

## Trained model directory
expname='10021600'
layer_name='leaky_re_lu_21'
model_basedir="/lus/theta-fs0/projects/CSC249ADCD01/clouds_tak/rotate_invariant/stepbystep/transformTF2"
model_datadir="${model_basedir}/output_model"

echo Num. of Core  per Node  == $cores_per_node
echo Num. of Nodes per Block == $nodes_per_block
cd $cwd

python ${cwd}/run_scale.py \
 --EXEC 'theta' \
 --queue ${queue} \
 --walltime ${walltime} \
 --stride 128  --patch_size 128 \
 --nclusters ${nclusters} \
 --num_cores $cores_per_node \
 --max_workers $max_workers  \
 --cores_per_worker $cores_per_worker \
 --nodes_per_block  $nodes_per_block \
 --tf_datadir $tf_datadir \
 --instrument $instrument \
 --hdf_mod02_datadir $hdf_mod02_datadir \
 --hdf_mod03_datadir $hdf_mod03_datadir \
 --hdf_mod06_datadir $hdf_mod06_datadir \
 --outputbasedir $outputbasedir \
 --centroids_filename $centroids_filename \
 --model_datadir $model_datadir \
 --expname ${expname} \
 --layer_name ${layer_name} \
 --ocean_only \
 --date ${date} \
 --ndays ${ndays}
 #--append   # add append option if you update already computed file to add new information

echo NORMAL END


