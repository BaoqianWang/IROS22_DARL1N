#!/bin/sh

# 6 agents
mpirun -n 4 python3  -m maddpg_o.experiments.train_darl1n \
    --scenario=grassland \
    --good-sight=0.2 \
    --adv-sight=100 \
    --num-agents=6 \
    --num-learners=3 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-landmark=3\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/grassland/darl1n/6agents/6agents_1" \
    --adv-load-dir="../result/grassland/baseline_maddpg/6agents/6agents_1" \
    --save-rate=30 \
    --prosp-dist=0.1 \
    --eva-max-episode-len=25 \
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --max-num-train=2000\
    --eva-max-episode-len=25 \
    --max-episode-len=25 \
    --ratio=1.5 \
    --seed=16\
    --load-one-side \


# 12 agents

# mpirun -n 7 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=grassland \
#     --good-sight=0.25 \
#     --adv-sight=100 \
#     --num-agents=12 \
#     --num-learners=6 \
#     --num-adversaries=6 \
#     --num-learners=6 \
#     --num-food=6 \
#     --num-landmark=6\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/grassland/darl1n/12agents/12agents_1" \
#     --adv-load-dir="../result/grassland/baseline_maddpg/12agents/12agents_1" \
#     --save-rate=30 \
#     --prosp-dist=0.15 \
#     --good-max-num-neighbors=12 \
#     --adv-max-num-neighbors=12 \
#     --max-num-train=3000\
#     --eva-max-episode-len=30 \
#     --max-episode-len=30 \
#     --ratio=2 \
#     --seed=16\
#     --batch-size=1024\
#     --load-one-side \
#
#
#
#
# # 24 agents
#
# mpirun -n 13 python3  -m maddpg_o.experiments.train_darl1n \
#   --scenario=grassland \
#   --good-sight=0.3 \
#   --adv-sight=100 \
#   --num-agents=24 \
#   --num-learners=12 \
#   --num-adversaries=12 \
#   --num-learners=12 \
#   --num-food=12 \
#   --num-landmark=12\
#   --good-policy=maddpg \
#   --adv-policy=maddpg \
#   --save-dir="../result/grassland/darl1n/24agents/24agents_1" \
#   --adv-load-dir="../result/grassland/baseline_maddpg/24agents/24agents_1" \
#   --save-rate=30 \
#   --prosp-dist=0.2 \
#   --eva-max-episode-len=35 \
#   --max-episode-len=35 \
#   --good-max-num-neighbors=24 \
#   --adv-max-num-neighbors=24 \
#   --max-num-train=3000\
#   --ratio=2.5 \
#   --seed=16\
#   --batch-size=1024\
#   --load-one-side \
#
#
#
#
#
#
# # 48 agents
# mpirun -n 25 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=grassland \
#     --good-sight=0.35 \
#     --adv-sight=100 \
#     --num-agents=48 \
#     --num-learners=24 \
#     --num-adversaries=24 \
#     --num-food=24 \
#     --num-landmark=24\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/grassland/darl1n/48agents/48agents_1" \
#     --adv-load-dir="../result/grassland/baseline_maddpg/48agents/48agents_1" \
#     --save-rate=30 \
#     --prosp-dist=0.05 \
#     --eva-max-episode-len=40 \
#     --max-episode-len=40 \
#     --good-max-num-neighbors=48 \
#     --adv-max-num-neighbors=48 \
#     --max-num-train=2000\
#     --eva-max-episode-len=40 \
#     --max-episode-len=40 \
#     --ratio=3 \
#     --seed=16\
#     --load-one-side \
