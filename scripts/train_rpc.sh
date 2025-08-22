#!/bin/bash
rlaunch --cpu=15 --gpu=2 --memory=74080 -- python3 main/train.py --cfg configs/rpc.yaml