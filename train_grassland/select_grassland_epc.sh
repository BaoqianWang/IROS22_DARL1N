#!/bin/ksh

# 6 Agents
python3 -m maddpg_o.experiments.train_epc_select \
    --scenario=grassland \
    --num-agents=6 \
    --num-good=3 \
    --num-adversaries=3 \
    --good-load-dir="../result/grassland/epc/6agents" \
    --save-dir="../result/grassland/epc/6agents" \
    --dir-ids 1 2 3\
    --num-selection=2


# 12 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=grassland \
#     --num-agents=12 \
#     --num-good=6 \
#     --num-adversaries=6 \
#     --good-load-dir="../result/grassland/epc/12agents" \
#     --save-dir="../result/grassland/epc/12agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 24 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=grassland \
#     --num-agents=24 \
#     --num-good=12 \
#     --num-adversaries=12 \
#     --good-load-dir="../result/grassland/epc/24agents" \
#     --save-dir="../result/grassland/epc/24agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 48 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=grassland \
#     --num-agents=48 \
#     --num-good=24 \
#     --num-adversaries=24 \
#     --good-load-dir="../result/grassland/epc/48agents" \
#     --save-dir="../result/grassland/epc/48agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2
