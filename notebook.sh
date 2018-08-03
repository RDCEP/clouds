
# This should work on gpu2 on midway

export KERAS_BACKEND=tensorflow

module purge

module load Anaconda3/5.0.0.1
module load graphviz
module load ImageMagick
module load cuda/9.0
source activate DL_CPU

ip=$(/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}')
port=$((10000+ $RANDOM % 20000))

echo "Starting ipython notebook ..."

jupyter notebook --no-browser --ip=$ip --port=$port
