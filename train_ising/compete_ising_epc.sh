#!/bin/sh

# 9 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=ising \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-good=9 \
    --last-adv=0 \
    --last-good=9 \
    --num-food=3\
    --num-agents=9 \
    --num-adversaries=0 \
    --max-num-train=250 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="../result/ising/epc/9agents/9agents_1" \
    --good-load-dir2="../result/ising/epc/9agents/9agents_1" \
    --save-dir="../result/ising/epc/9agents/9agents_1" \
    --save-rate=0 \
    --train-rate=0 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16 \
    --batch-size=32 \
    --restore \



# 16 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000 \
#     --num-agents=16 \
#     --num-good=16 \
#     --last-adv=0 \
#     --last-good=16 \
#     --num-food=6 \
#     --num-adversaries=0 \
#     --max-num-train=100 \
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/ising/epc/16agents/16agents_1" \
#     --good-load-dir2="../result/ising/epc/16agents/16agents_1" \
#     --save-dir="../result/ising/epc/16agents/16agents_1" \
#     --save-rate=0\
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --timeout=0.3 \
#     --seed=16 \
#     --batch-size=32 \
#     --restore

# 25 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000 \
#     --num-agents=25 \
#     --num-good=25 \
#     --last-adv=0 \
#     --last-good=25 \
#     --num-food=6 \
#     --num-adversaries=0 \
#     --max-num-train=100 \
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/ising/epc/25agents/25agents_1" \
#     --good-load-dir2="../result/ising/epc/25agents/25agents_1" \
#     --save-dir="../result/ising/epc/25agents/25agents_1" \
#     --save-rate=0 \
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --timeout=0.3 \
#     --seed=16 \
#     --batch-size=32 \
#     --restore \


# 64 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=ising \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000 \
#     --num-agents=64\
#     --num-good=64 \
#     --last-adv=0 \
#     --last-good=64\
#     --num-food=64 \
#     --num-adversaries=0 \
#     --max-num-train=100 \
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/ising/epc/64agents/64agents_1" \
#     --good-load-dir2="../result/ising/epc/64agents/64agents_1" \
#     --save-dir="../result/ising/epc/64agents/64agents_1" \
#     --save-rate=0 \
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --timeout=0.3 \
#     --seed=16 \
#     --batch-size=32 \
#     --restore \
