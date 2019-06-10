#!/bin/bash

#SBATCH --account=pi-foster  # default pi-chard
#SBATCH --job-name=spark_bkmeans
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=broadwl   #28 cores 64GB memory
##SBATCH --partition=broadwl-lc   #28 cores 64GB memory
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --array=16   # number of cluster


module load spark/2.3.0
module load Anaconda3/5.0.0.1
source activate tf-cpu

which python

##### params
ncore=4          #default:  580  for 80nodes
driver_mem=20    #default:  100G for 80nodes
exec_mem=220     #default: 4800G for 80nodes

##### setting I
scale_patch_size=1500  # unit[K-patches] ==>  1,000 patches
# e.g. current maximum is about 1,500. 1,700 will be OOM error

##### settnig II
bands="28_29_31"
inputdir=`pwd`/output_clouds_feature_2000_2018
outputdir=`pwd`/output_clustering_2000_2018_m01_band${bands}
model_name='m01_b'${bands}

spark-submit --total-executor-cores ${ncore}   \
             --driver-memory ${driver_mem}G \
             --executor-memory ${exec_mem}G \
             --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:G1HeapRegionSize=32" \
             prg_pyspark_m02_bisectingKmeans.py \
              --scale_patch_size=${scale_patch_size}  \
              --n_cluster=${SLURM_ARRAY_TASK_ID} \
              --inputdir=${inputdir} \
              --outputdir=${outputdir} \
              --oname=2000-2018_random \
              --modelname=${model_name}

#--oname=2000-2018 \

### setting examples

#**** nodes=5
#ncore=100         #default:  580  for 80nodes
#driver_mem=10    #default:  100G for 80nodes
#exec_mem=3000     #default: 4800G for 80nodes
#scale_patch_size=~100  # unit[K-patches] ==>  1,000 patches

#**** nodes=20
#ncore=200         #default:  580  for 80nodes
#driver_mem=50    #default:  100G for 80nodes
#exec_mem=1000     #default: 4800G for 80nodes
#scale_patch_size=~500  # unit[K-patches] ==>  1,000 patches


#**** nodes=80 - 100
#ncore=580         #default:  580  for 80nodes
#driver_mem=100-200    #default:  100G for 80nodes
#exec_mem=4800(80)-6000(100)     #default: 4800G for 80nodes
#scale_patch_size=~1500  # unit[K-patches] ==>  1,000 patches


# Good version
# memory option shoulf be specified o.w. OOM kill
# This code works for scale-size = 20 case
#spark-submit --total-executor-cores 280   \
#             --executor-memory 500G \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=20
# node = 80
#spark-submit --total-executor-cores 560   \
#             --driver-memory 100G \
#             --executor-memory 4800G \
#             --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:G1HeapRegionSize=32" \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=15
#
# * ==> This script is applicable to --scal_size = 10, 5 and 2


# machine ==> broadwl
# failed settings
# node = 20
#spark-submit --total-executor-cores 500   \
#             --executor-memory 1000G \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=15
# node = 40
#spark-submit --total-executor-cores 500   \
#             --executor-memory 2200G \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=15

# node = 50
#spark-submit --total-executor-cores 500   \
#             --executor-memory 3000G \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=15

# node = 80
#spark-submit --total-executor-cores 560   \
#             --driver-memory 100G \
#             --executor-memory 4800G \
#             --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:G1HeapRegionSize=32" \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=1

# node = 90 : cluster = 10 Bus Error
#spark-submit --total-executor-cores 560   \
#             --driver-memory 200G \
#             --executor-memory 5200G \
#             --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:G1HeapRegionSize=32" \
#             prg_pyspark_read_patches_nclusters.py \
#              --scale_size=3  --n_cluster=${SLURM_ARRAY_TASK_ID} \
#              --inputdir=${inputdir} \
#              --oname=2015July


# node = 100
#spark-submit --total-executor-cores 560   \
#             --driver-memory 250G \
#             --executor-memory 6100G \
#             --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:G1HeapRegionSize=32" \
#             prg_pyspark_read_patches_halfsize.py  --scale_size=1 - 1.8
