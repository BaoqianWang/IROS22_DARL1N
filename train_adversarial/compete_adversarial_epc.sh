#!/bin/sh

# 6 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000 \
#     --num-good=3 \
#     --last-adv=3 \
#     --last-good=3 \
#     --num-food=3\
#     --num-agents=6 \
#     --num-adversaries=3 \
#     --max-num-train=1000 \
#     --ratio=1.5 \
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/adversarial/epc/6agents/6agents_1" \
#     --good-load-dir2="../result/adversarial/epc/6agents/6agents_1" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/6agents/6agents_1" \
#     --save-dir="../result/adversarial/epc/6agents/6agents_1" \
#     --save-rate=0 \
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --timeout=0.3 \
#     --seed=16 \
#     --adv-load-one-side \
#     --restore

# 12 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000 \
#     --num-agents=12\
#     --num-good=6 \
#     --last-adv=6 \
#     --last-good=6\
#     --num-food=6 \
#     --num-adversaries=6 \
#     --max-num-train=2000 \
#     --ratio=2.0 \
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/adversarial/epc/12agents/12agents_1" \
#     --good-load-dir2="../result/adversarial/epc/12agents/12agents_1" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/12agents/12agents_1" \
#     --save-dir="../result/adversarial/epc/12agents/12agents_1" \
#     --save-rate=0 \
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --max-episode-len=30 \
#     --timeout=0.3 \
#     --seed=16 \
#     --adv-load-one-side \
#     --restore \


# 24 agents
# python3 -m maddpg_o.experiments.train_epc \
#     --scenario=adversarial \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-episodes=1000\
#     --num-good=12\
#     --num-agents=24\
#     --last-adv=12\
#     --last-good=12\
#     --num-food=12\
#     --num-adversaries=12\
#     --max-num-train=600\
#     --ratio=2.5\
#     --num-units=64 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --good-load-dir1="../result/adversarial/epc/24agents/24agents_1" \
#     --good-load-dir2="../result/adversarial/epc/24agents/24agents_1" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/24agents/24agents_1" \
#     --save-dir="../result/adversarial/epc/24agents/24agents_1" \
#     --save-rate=0 \
#     --train-rate=0 \
#     --n-cpu-per-agent=5 \
#     --n-envs=25 \
#     --max-episode-len=35 \
#     --timeout=0.3 \
#     --seed=16 \
#     --adv-load-one-side \
#     --restore \

# 48 agents
python3 -m maddpg_o.experiments.train_epc \
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-episodes=1000\
    --num-good=24\
    --num-agents=48\
    --last-adv=24\
    --last-good=24\
    --num-food=24\
    --num-adversaries=24\
    --max-num-train=200\
    --ratio=3\
    --num-units=64 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --good-load-dir1="../result/adversarial/epc/24agents/48agents_1" \
    --good-load-dir2="../result/adversarial/epc/24agents/48agents_1" \
    --adv-load-dir="../result/adversarial/baseline_maddpg/48agents/48agents_1" \
    --save-dir="../result/adversarial/epc/48agents/48agents_1" \
    --save-rate=0 \
    --train-rate=0 \
    --n-cpu-per-agent=5 \
    --n-envs=25 \
    --max-episode-len=40 \
    --timeout=0.3 \
    --seed=16 \
    --adv-load-one-side \
    --restore \
