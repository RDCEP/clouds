#!/bin/bash

python3 prg_geturl2.py --download_process='combining_fn' \
    --outputdir=$(pwd) \
    --processors=7 \
    --keyword='MOD35' \
    --thresval=1 \
    --days=1 \
    --start='2000-02-24'\
    --end='2019-03-27'\
    --datedata='clustering_mod35_list.txt'\
