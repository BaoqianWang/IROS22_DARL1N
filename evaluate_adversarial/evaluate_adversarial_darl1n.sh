#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=adversarial \
    --good-sight=0.2 \
    --adv-sight=100 \
    --num-agents=6 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-landmark=3 \
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --good-save-dir="../result/adversarial/darl1n/6agents/6agents_eva" \
    --adv-save-dir="../result/adversarial/baseline_maddpg/6agents/6agents_eva" \
    --save-rate=100 \
    --train-rate=100 \
    --prosp-dist=0.1 \
    --ratio=1.5 \
    --max-episode-len=25 \
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --method="darl1n" \
    --display \
