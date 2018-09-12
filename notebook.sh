
# This should work on gpu2 on midway

export KERAS_BACKEND=tensorflow

module purge

module load Anaconda3/5.0.0.1
module load cuda/9.0
# TODO flags for spark mode?
# module load spark
# source activate cc2
source activate clouds


ip=$(/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}')
port=$((10000+ $RANDOM % 20000))
jupyter notebook --no-browser --ip=$ip --port=$port
