#!/bin/ksh

# 3 Agents
python3 -m maddpg_o.experiments.train_epc_select \
    --scenario=simple_spread \
    --num-agents=3 \
    --num-good=3 \
    --num-adversaries=0 \
    --good-load-dir="../result/simple_spread/epc/3agents/3agents" \
    --save-dir="../result/simple_spread/epc/3agents" \
    --dir-ids 1 2 3\
    --num-selection=2


# 6 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=simple_spread \
#     --num-agents=6 \
#     --num-good=6 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/simple_spread/epc/6agents/6agents" \
#     --save-dir="../result/simple_spread/epc/6agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 12 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=simple_spread \
#     --num-agents=12 \
#     --num-good=12 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/simple_spread/epc/12agents/12agents" \
#     --save-dir="../result/simple_spread/epc/12agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 24 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=simple_spread \
#     --num-agents=24 \
#     --num-good=24 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/simple_spread/epc/24agents/24agents" \
#     --save-dir="../result/simple_spread/epc/24agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2
