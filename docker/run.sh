#!/bin/bash
set -e
set -u


export DISPLAY=$DISPLAY
echo "setting display to $DISPLAY"
xhost +
docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name=isaacgym_container -v $HOME/gymworkspace:/gymworkspace:rw isaacgym /bin/bash
xhost -
fi
