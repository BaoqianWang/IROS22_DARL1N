#!/bin/sh
# Training script for running darl1n on Ising Model



# 9 agents
mpirun -n 10 python3  -m maddpg_o.experiments.train_darl1n \
    --scenario=ising \
    --good-sight=100 \
    --adv-sight=100 \
    --num-agents=9 \
    --num-learners=9  \
    --num-adversaries=0 \
    --num-food=3 \
    --num-landmark=3\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/ising/darl1n/9agents/9agents_1/" \
    --save-rate=10 \
    --prosp-dist=0.3 \
    --max-num-train=150 \
    --good-max-num-neighbors=5 \
    --adv-max-num-neighbors=5 \
    --eva-max-episode-len=25 \
    --batch-size=32 \
    --seed=16 \

# 16agents
# mpirun -n 17 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --good-sight=100 \
#     --adv-sight=100 \
#     --num-agents=16 \
#     --num-learners=16  \
#     --num-adversaries=0 \
#     --num-food=3 \
#     --num-landmark=3\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/16agents/16agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=150 \
#     --good-max-num-neighbors=5 \
#     --adv-max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \


# 25 agents
# mpirun -n 26 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --good-sight=100 \
#     --adv-sight=100 \
#     --last-good=16 \
#     --last-adv=0 \
#     --num-agents=25 \
#     --num-adversaries=0 \
#     --num-learners=25 \
#     --num-food=3 \
#     --num-landmark=3\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/25agents/25agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=300 \
#     --good-max-num-neighbors=5 \
#     --adv-max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \


# 64 agents
# mpirun -n 65 python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --sight=0.5 \
#     --num-agents=64 \
#     --last-stage-num=16 \
#     --num-adversaries=0 \
#     --num-food=3 \
#     --num-landmark=3\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/64agents/64agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=100 \
#     --max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \
