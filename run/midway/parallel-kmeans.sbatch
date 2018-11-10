#!/bin/sh
#SBATCH --job-name      pkmeans
#SBATCH --output        logs/midway/%x_%A.out
#SBATCH --partition     broadwl
#SBATCH --ntasks        16
#SBATCH --mem-per-cpu   32000
#SBATCH --account       pi-chard
#SBATCH --time          04:00:00

# The kmeans implementation is from `http://www.ece.northwestern.edu/~wkliao/Kmeans/index.html`
# the $POINTS file is first `n: int32`, `d: int32` which are number of vectors then dimension
# after that its n * d float32 numbers in the vectors in C order.

# The initial cluster centers are the first $K vectors so you should permute them with numpy
# to get different results on the same set of points. These can be compared with mutual information
# if you keep the permutation.

K=60
POINTS=/project/foster/clouds/output/mod09cnn15b/encodings/m15b-enc2
mpiexec -n $SLURM_NTASKS ./parallel-kmeans/mpi_main -o -b -i $POINTS -n $K 

