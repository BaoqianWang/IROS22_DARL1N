#!/bin/sh

mpirun -n 2 \
python3  -m maddpg_o.experiments.train_darl1n \
    --scenario=simple_spread \
    --good-sight=0.15 \
    --adv-sight=100.0 \
    --num-agents=1 \
    --num-learners=1 \
    --num-adversaries=0 \
    --num-food=1 \
    --num-landmark=1\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/simple_spread/darl1n/1agents/1agents_1/" \
    --save-rate=30 \
    --max-num-train=6000\
    --prosp-dist=0.05 \
    --eva-max-episode-len=25 \
    --good-max-num-neighbors=1 \
    --adv-max-num-neighbors=1 \
    --ratio=1 \
    --seed=16\

# mpirun -n 4 \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=simple_spread \
#     --good-sight=0.15 \
#     --adv-sight=100.0 \
#     --num-agents=3 \
#     --num-learners=3 \
#     --num-adversaries=0 \
#     --num-food=3 \
#     --num-landmark=3\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/darl1n/3agents/3agents_1/" \
#     --save-rate=30 \
#     --max-num-train=3000\
#     --prosp-dist=0.05 \
#     --eva-max-episode-len=25 \
#     --good-max-num-neighbors=3 \
#     --adv-max-num-neighbors=3 \
#     --ratio=1 \
#     --seed=16\


# # 6 agents
# mpirun -n 7 \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=simple_spread \
#     --good-sight=0.2 \
#     --adv-sight=100.0 \
#     --num-agents=6 \
#     --num-learners=6\
#     --num-adversaries=0 \
#     --num-food=6 \
#     --num-landmark=6\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/darl1n/6agents/6agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000\
#     --prosp-dist=0.1 \
#     --eva-max-episode-len=25 \
#     --good-max-num-neighbors=6 \
#     --adv-max-num-neighbors=6 \
#     --ratio=1.5 \
#     --seed=16\


# # 12 agents
# mpirun -n 13 \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=simple_spread \
#     --good-sight=0.25 \
#     --adv-sight=100.0 \
#     --num-agents=12 \
#     --num-learners=12\
#     --num-adversaries=0 \
#     --num-food=12 \
#     --num-landmark=12\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/darl1n/12agents/12agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000\
#     --prosp-dist=0.15 \
#     --eva-max-episode-len=25 \
#     --good-max-num-neighbors=12 \
#     --adv-max-num-neighbors=12 \
#     --ratio=2.0 \
#     --seed=16\


# # 24 agents
#
# mpirun -n 25 \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=simple_spread \
#     --good-sight=0.3 \
#     --adv-sight=100.0 \
#     --num-agents=24 \
#     --num-learners=24\
#     --num-adversaries=0 \
#     --num-food=24 \
#     --num-landmark=24\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/simple_spread/darl1n/24agents/24agents_1/" \
#     --save-rate=30 \
#     --max-num-train=2000\
#     --prosp-dist=0.2 \
#     --eva-max-episode-len=25 \
#     --good-max-num-neighbors=24 \
#     --adv-max-num-neighbors=24 \
#     --ratio=2.5 \
#     --seed=16\
