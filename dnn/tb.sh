#!/bin/bash
if [ "$#" -eq 1 ]; then
    logdir=$1
    ip=$(/sbin/ip route get 8.8.8.8 | awk '{print $NF;exit}')
    port=$((10000+ $RANDOM % 20000))

    echo opening tensorboard at $ip:$port

    tensorboard --logdir $logdir --port $port

else
    echo provide log dir
fi

