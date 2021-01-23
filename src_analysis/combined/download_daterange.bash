#!/bin/bash

cwd=`pwd`

python3 prg_geturl3.py --download_process='combining_fn' \
    --outputdir="/home/tkurihana/Research/data/MYD35" \
    --processors=4 \
    --keyword='MYD35' \
    --thresval=1 \
    --days=1 \
    --start='2000-02-24'\
    --end='2019-03-27'\
    --datedata='sample.txt'\
    --appkey='210B0BEC-13E1-11EA-B059-AE780D77E571'
    #--datedata='clustering_mod35_list.txt'\
    #--outputdir=${cwd}/MOD35 \
