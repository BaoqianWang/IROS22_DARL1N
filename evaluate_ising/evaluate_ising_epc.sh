#!/bin/sh

python3  -m maddpg_o.experiments.evaluate_epc \
--scenario=ising \
--good-sight=100.0 \
--adv-sight=100.0 \
--num-good=9 \
--num-adversaries=0 \
--num-food=12 \
--checkpoint-rate=0 \
--good-policy=att-maddpg \
--adv-policy=att-maddpg \
--good-share-weights \
--adv-share-weights \
--num-units=64 \
--good-save-dir="../result/ising/epc/9agents/9agents_eva/" \
