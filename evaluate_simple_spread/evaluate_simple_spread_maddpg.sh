#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_normal \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-adversaries=0 \
    --num-food=1 \
    --num-agents=1 \
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --good-save-dir="../result/simple_spread/maddpg/1agents/1agents_1/" \
    --save-rate=30 \
    --good-max-num-neighbors=1 \
    --adv-max-num-neighbors=1 \
    --method="maddpg" \
    --ratio=1 \
    --display \



# python3  -m maddpg_o.experiments.evaluate_normal \
#     --scenario=simple_spread\
#     --good-sight=100 \
#     --adv-sight=100 \
#     --num-agents=3 \
#     --num-adversaries=0 \
#     --num-food=3 \
#     --num-landmark=3 \
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --good-save-dir="../result/simple_spread/maddpg/3agents/3agents_eva/"\
#     --save-rate=100 \
#     --train-rate=100 \
#     --max-episode-len=25 \
#     --good-max-num-neighbors=3 \
#     --adv-max-num-neighbors=3 \
#     --method="maddpg" \
#     --ratio=1 \
#     --display \
