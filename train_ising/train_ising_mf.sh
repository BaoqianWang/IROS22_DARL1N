#!/bin/sh

# 9 agents
python3 -m maddpg_o.experiments.train_normal \
    --scenario=ising \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-agents=9 \
    --num-adversaries=0 \
    --num-food=9 \
    --good-policy=mean_field \
    --adv-policy=mean_field \
    --save-dir="../result/ising/mean_field/9agents/9agents_1" \
    --save-rate=10 \
    --max-num-train=250\
    --good-max-num-neighbors=9 \
    --adv-max-num-neighbors=9 \
    --seed=16

# 16 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-agents=16 \
#     --num-adversaries=0 \
#     --num-food=9 \
#     --good-policy=mean_field \
#     --adv-policy=mean_field \
#     --save-dir="../result/ising/mean_field/16agents/16agents_1" \
#     --save-rate=10 \
#     --max-num-train=250\
#     --good-max-num-neighbors=16 \
#     --adv-max-num-neighbors=16 \
#     --seed=16

## 25 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-agents=25 \
#     --num-adversaries=0 \
#     --num-food=9 \
#     --good-policy=mean_field \
#     --adv-policy=mean_field \
#     --save-dir="../result/ising/mean_field/25agents/25agents_1" \
#     --save-rate=10 \
#     --max-num-train=150\
#     --good-max-num-neighbors=25 \
#     --adv-max-num-neighbors=25 \
#     --seed=16 \
#
#
# # 64 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-agents=64\
#     --num-adversaries=0 \
#     --num-food=9 \
#     --good-policy=mean_field \
#     --adv-policy=mean_field \
#     --save-dir="../result/ising/mean_field/64agents/64agents_1" \
#     --save-rate=10 \
#     --max-num-train=250\
#     --good-max-num-neighbors=64 \
#     --adv-max-num-neighbors=64 \
#     --seed=16
