#!/bin/sh

# 6 agents
python3 -m maddpg_o.experiments.train_normal \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-agents=6\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/adversarial/baseline_maddpg/6agents/6agents_1/" \
    --save-rate=30 \
    --max-num-train=100\
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --ratio=1.5 \
    --seed=16 \



# 12 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=6 \
#     --num-food=6 \
#     --num-agents=12\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/baseline_maddpg/12agents/12agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000\
#     --good-max-num-neighbors=12 \
#     --adv-max-num-neighbors=12 \
#     --max-episode-len=30 \
#     --ratio=2 \
#     --seed=16 \


# 24 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=12 \
#     --num-food=12 \
#     --num-agents=24\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/baseline_maddpg/24agents/24agents_1/" \
#     --save-rate=30 \
#     --max-num-train=600\
#     --good-max-num-neighbors=24 \
#     --adv-max-num-neighbors=24 \
#     --max-episode-len=35 \
#     --ratio=2.5 \
#     --seed=16

# 48 agents
# python3 -m maddpg_o.experiments.train_normal \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-adversaries=24 \
#     --num-food=24 \
#     --num-agents=48\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/baseline_maddpg/48agents/48agents_1/" \
#     --save-rate=30 \
#     --max-num-train=100\
#     --good-max-num-neighbors=48 \
#     --adv-max-num-neighbors=48 \
#     --max-episode-len=40 \
#     --ratio=3 \
#     --seed=16 \
