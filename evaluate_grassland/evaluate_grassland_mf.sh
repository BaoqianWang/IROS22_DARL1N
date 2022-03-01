#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=grassland\
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-agents=6 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-landmark=3 \
    --good-policy=mean_field \
    --adv-policy=maddpg \
    --good-save-dir="../result/grassland/mean_field/6agents/6agents_eva/"\
    --adv-save-dir="../result/grassland/baseline_maddpg/6agents/6agents_eva/"\
    --save-rate=100 \
    --train-rate=100 \
    --max-episode-len=25 \
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --method="mean_field" \
    --ratio=1.5 \
    --display \
