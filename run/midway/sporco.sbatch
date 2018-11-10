#!/bin/sh
#SBATCH --job-name=sporco
#SBATCH --output=sporco-array/%x-%A-%a.out
#SBATCH --partition=broadwl
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu=1000
#SBATCH --account=pi-chard
#SBATCH --time=00:10:00
#SBATCH --array 0-49%10

module purge
module load Anaconda3
source activate cc4

PARAMS=(`python -c """
import numpy as np
import sys
l1s = np.logspace(1, 2, 10)
mults = np.logspace(-2, 2, 5, base=2)
i = int(sys.argv[1])
print(l1s[ i // 5 ], mults[i % 5])
""" $SLURM_ARRAY_TASK_ID`)


L1_REG=${PARAMS[0]}
D_MULT=${PARAMS[1]}
ENCODINGS=/project/foster/clouds/output/mod09cnn15b/encodings/m15b-enc-flat
OUT_DIR=/project/foster/clouds/sporco/m15b-n$L1_REG-d$D_MULT/

echo $L1_REG, $D_MULT
echo $ENCODINGS
echo $OUT_DIR

python experimental/sparse_encode.py \
    $ENCODINGS $OUT_DIR \
    --num_iterations 100 \
    --max_vecs 50000 \
    --l1_regularization $L1_REG \
    --code_mult $D_MULT
