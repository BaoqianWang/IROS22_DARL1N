#!/bin/sh

# python3 -m maddpg_o.experiments.evaluate_epc\
#     --scenario=simple_spread \
#     --good-sight=100.0 \
#     --adv-sight=100.0 \
#     --num-good=3 \
#     --num-adversaries=0 \
#     --num-food=3 \
#     --num-agents=3 \
#     --checkpoint-rate=0 \
#     --good-policy=att-maddpg \
#     --adv-policy=att-maddpg \
#     --good-share-weights \
#     --adv-share-weights \
#     --num-units=64 \
#     --ratio=1 \
#     --good-save-dir="../result/simple_spread/epc/3agents/3agents_eva/" \

python3 -m maddpg_o.experiments.evaluate_epc\
    --scenario=simple_spread \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-good=12 \
    --num-adversaries=0 \
    --num-food=12 \
    --num-agents=12 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --num-units=64 \
    --ratio=2 \
    --good-save-dir="../result/simple_spread/epc/12agents_1/" \
