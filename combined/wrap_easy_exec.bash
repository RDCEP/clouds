#!/bin/bash

appkey='210B0BEC-13E1-11EA-B059-AE780D77E571'

BASEDIR="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD021KM"
#outputdir="/home/tkurihana/Research/data/MYD02"
outputdir="/home/tkurihana/scratch-midway2/data/MYD02"

#dates_list=(
#)


#for idate in "${dates_list[@]}" ; do
#  echo $idate
#  inputdir=$BASEDIR/$iyear/$idate/
#  bash easy_execute.bash $inputdir $appkey $outputdir
#done

while read line
do 
  iyear=`echo ${line:0:4}`
  idate=`echo ${line:4}`
  echo $iyear $idate

  inputdir=$BASEDIR/$iyear/$idate/
  bash easy_execute.bash $inputdir $appkey $outputdir
done  < ./ocean_train.txt
