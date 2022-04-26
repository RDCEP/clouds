#!/bin/bash

# Parameter to run aicca run app
clusterstamp="2003-015-105-195-288_053-132-185-315" # timestamp of cluster centroids
year=2003 
instrument='terra'
nclusters=42
## Number of additional days 
ndays=6 #  6 means running file + 6 days more 

## Current work directory
cwd=`pwd`

##############################################
### Main run
##############################################
AICCADAYS=()

sday=1
ncount=1
for ielem in `seq 1 1 ${ncount}` ; do
  AICCADAYS+=($sday)
  sday=`expr ${sday} + ${ndays} + 1`
  echo $sday
done


for i in "${AICCADAYS[@]}" ; do
  idate=`printf %03d $i`
  echo $idate

  # copy 
  mkdir -p  ./${clusterstamp}/workspace-${year}-${idate}
  cd ./${clusterstamp}/workspace-${year}-${idate}
  cp -r ${cwd}/exec.bash .
  cp -r ${cwd}/run.py .
  cp -r ${cwd}/run_scale.py .

  chmod +x exec.bash

  wwd=`pwd`  
  nohup bash exec.bash $year $idate $nclusters ${clusterstamp} ${wwd}  ${ndays}  ${instrument} > ${year}-${idate}.out &
  cd $cwd
done
