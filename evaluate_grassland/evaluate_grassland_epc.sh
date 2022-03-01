#!/bin/sh

python3 -m maddpg_o.experiments.evaluate_epc\
    --scenario=grassland \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-good=3 \
    --num-adversaries=3 \
    --num-food=3 \
    --checkpoint-rate=0 \
    --num-agents=6 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --num-units=64 \
    --ratio=1.5 \
    --max-episode-len=25 \
    --adv-save-dir="../result/grassland/baseline_maddpg/6agents/6agents_eva/" \
    --good-save-dir="../result/grassland/epc/6agents/6agents_eva/" \
