#!/bin/bash
rlaunch --cpu=32 --gpu=4 --memory=144080 --preemptible=no -- python3 main/visual.py --cfg configs/rpc_hourglass_visual.yaml