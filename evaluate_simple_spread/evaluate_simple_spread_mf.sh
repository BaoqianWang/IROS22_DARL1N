#!/bin/sh



python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=simple_spread\
    --good-sight=100 \
    --adv-sight=100 \
    --num-agents=3 \
    --num-adversaries=0 \
    --num-food=3 \
    --num-landmark=3 \
    --good-policy=mean_field \
    --adv-policy=mean_field \
    --good-save-dir="../result/simple_spread/mean_field/3agents/3agents_eva/"\
    --save-rate=100 \
    --train-rate=100 \
    --max-episode-len=25 \
    --good-max-num-neighbors=3 \
    --adv-max-num-neighbors=3 \
    --method="mean_field" \
    --ratio=1 \
    --display \
