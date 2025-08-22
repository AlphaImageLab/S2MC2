#!/bin/bash
rlaunch --cpu=20 --gpu=4 --memory=144080 -- python3 main/train.py --cfg configs/rpc_hourglass_same.yaml