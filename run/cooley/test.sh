#!/bin/bash
#COBALT -t 5 -n 1 -O test


source activate clouds
echo "Starting"
./experimental/test/test.sh
