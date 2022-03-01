#!/bin/ksh

# 9 Agents
python3 -m maddpg_o.experiments.train_epc_select \
    --scenario=ising \
    --num-agents=9 \
    --num-good=9 \
    --num-adversaries=0 \
    --good-load-dir="../result/ising/epc/9agents" \
    --save-dir="../result/ising/epc/9agents" \
    --dir-ids 1 2 3\
    --num-selection=2


# 16 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=ising \
#     --num-agents=16 \
#     --num-good=16 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/ising/epc/16agents" \
#     --save-dir="../result/ising/epc/16agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 25 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=ising \
#     --num-agents=25 \
#     --num-good=25 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/ising/epc/25agents" \
#     --save-dir="../result/ising/epc/25agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2


# 64 Agents
# python3 -m maddpg_o.experiments.train_epc_select \
#     --scenario=ising \
#     --num-agents=64 \
#     --num-good=64 \
#     --num-adversaries=0 \
#     --good-load-dir="../result/ising/epc/64agents" \
#     --save-dir="../result/ising/epc/64agents" \
#     --dir-ids 1 2 3\
#     --num-selection=2
