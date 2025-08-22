#!/bin/bash
rlaunch --cpu=20 --gpu=8 --memory=144080 -- python3 main/train.py --cfg configs/rpc_hourglass_adv_pretrain.yaml