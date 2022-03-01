#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=grassland \
    --good-sight=0.2 \
    --adv-sight=100 \
    --num-agents=6 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-landmark=3 \
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --good-save-dir="../result/grassland/darl1n/6agents/6agents_eva" \
    --adv-save-dir="../result/grassland/baseline_maddpg/6agents/6agents_eva" \
    --save-rate=100 \
    --train-rate=100 \
    --prosp-dist=0.1 \
    --max-episode-len=25 \
    --ratio=1.5 \
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --method="darl1n" \
    --display \
