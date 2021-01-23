#!/bin/bash

appkey='210B0BEC-13E1-11EA-B059-AE780D77E571'

BASEDIR="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD021KM/2008"
outputdir="/home/tkurihana/Research/data/MYD02"

dates_list=(
  '053' '068' '121' '122' '168' '189' '235' '272' '282' '312' '337'
)


for idate in "${dates_list[@]}" ; do
  echo $idate

  inputdir=$BASEDIR/$idate/
  bash easy_execute.bash $inputdir $appkey $outputdir
done
