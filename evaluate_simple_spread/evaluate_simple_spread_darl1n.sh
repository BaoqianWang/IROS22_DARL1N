#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=simple_spread\
    --good-sight=0.15 \
    --adv-sight=100 \
    --num-agents=1 \
    --num-adversaries=0 \
    --num-food=1 \
    --num-landmark=1\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --good-save-dir="../result/simple_spread/darl1n/1agents/1agents_1/" \
    --save-rate=100 \
    --train-rate=100 \
    --ratio=1 \
    --max-episode-len=25 \
    --method="darl1n" \
    --good-max-num-neighbors=1 \
    --adv-max-num-neighbors=1 \
    --display \
