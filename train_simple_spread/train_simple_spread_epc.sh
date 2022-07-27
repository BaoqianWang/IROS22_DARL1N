#!/bin/ksh

# 3 Agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-good=3 \
    --num-food=3 \
    --num-adversaries=0 \
    --max-num-train=3000 \
    --ratio=1 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --save-dir="../result/simple_spread/epc/3agents/3agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16


# 6 Agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-agents=6 \
    --last-adv=0\
    --last-good=3\
    --num-good=6 \
    --num-food=6 \
    --num-adversaries=0 \
    --max-num-train=2000 \
    --ratio=1.5 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="../result/simple_spread/epc/3agents/3agents_eva" \
    --good-load-dir2="../result/simple_spread/epc/3agents/3agents_eva" \
    --save-dir="../result/simple_spread/epc/6agents/6agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16 \
    --restore \


# 12 Agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000 \
    --num-good=12 \
    --last-adv=0 \
    --last-good=6 \
    --num-food=12 \
    --num-adversaries=0 \
    --max-num-train=1000 \
    --ratio=2.0 \
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="../result/simple_spread/epc/6agents/6agents_1" \
    --good-load-dir2="../result/simple_spread/epc/6agents/6agents_1" \
    --save-dir="../result/simple_spread/epc/12agents/12agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16 \
    --restore \


# 24 Agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000\
    --num-good=24\
    --last-adv=0\
    --last-good=12\
    --num-food=24\
    --num-adversaries=0\
    --max-num-train=600\
    --ratio=2.5\
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="../result/simple_spread/epc/12agents/12agents_1" \
    --good-load-dir2="../result/simple_spread/epc/12agents/12agents_1" \
    --save-dir="../result/simple_spread/epc/24agents/24agents_1" \
    --save-rate=30 \
    --train-rate=100 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --timeout=0.3 \
    --seed=16 \
    --restore \
