#!/bin/bash
ROOT=/data/rpc/retail_product_checkout
rlaunch --cpu=16 --memory=4080 -- python3 tools/make_nori.py \
--root ${ROOT}/train2019/ \
--json_path ${ROOT}/instances_train2019.json \
--save_root ${ROOT}/nori/train2019/ \
--worker_num 16 \
--nori_name nori_single