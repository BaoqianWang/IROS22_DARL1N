#!/bin/sh

# 3 agents
python3 -m maddpg_o.experiments.train_normal \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-adversaries=0 \
    --num-food=3 \
    --num-agents=3 \
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/simple_spread/maddpg/3agents/3agents_1/" \
    --save-rate=30 \
    --max-num-train=2000 \
    --good-max-num-neighbors=3 \
    --adv-max-num-neighbors=3 \
    --seed=16 \
    --ratio=1 \

# 6 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=simple_spread \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=0 \
#     --num-food=6 \
#     --num-agents=6 \
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/maddpg/6agents/6agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000 \
#     --good-max-num-neighbors=6 \
#     --adv-max-num-neighbors=6 \
#     --seed=16 \
#     --ratio=1.5 \

# 12 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=simple_spread \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=0 \
#     --num-food=12 \
#     --num-agents=12 \
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/maddpg/12agents/12agents_1/" \
#     --save-rate=30 \
#     --max-num-train=1000 \
#     --good-max-num-neighbors=12 \
#     --adv-max-num-neighbors=12 \
#     --seed=16 \
#     --ratio=2 \
#
#
# 24 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=simple_spread \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=0 \
#     --num-food=24 \
#     --num-agents=24 \
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/maddpg/24agents/24agents_1/" \
#     --save-rate=30 \
#     --max-num-train=400 \
#     --good-max-num-neighbors=24 \
#     --adv-max-num-neighbors=24 \
#     --seed=16 \
#     --ratio=2.5 \
