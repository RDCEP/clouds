#!/bin/bash

python3 api_request.py --email='koenig1@uchicago.edu' \
    --app_key='126AA2A4-96BA-11E9-9D2C-D7883D88392C' \
    --products='MOD35_L2' \
    --date_file='label1.txt' \
    --coords_file='coords.csv' \
    --desired_dir='/Users/katykoeing/Desktop/clouds/src_analysis/api-requests/hdf_files' \
    --addl_info=False \
