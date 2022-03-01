#!/bin/sh


python3 -m maddpg_o.experiments.evaluate_epc\
    --scenario=adversarial \
    --good-sight=100.0 \
    --adv-sight=100.0 \
    --num-good=3 \
    --num-adversaries=3 \
    --num-food=3 \
    --checkpoint-rate=0 \
    --good-policy=att-maddpg \
    --adv-policy=att-maddpg \
    --good-share-weights \
    --adv-share-weights \
    --num-units=64 \
    --ratio=1.5 \
    --max-episode-len=25 \
    --good-save-dir="../result/adversarial/epc/6agents/6agents_eva/" \
    --adv-save-dir="../result/adversarial/baseline_maddpg/6agents/6agents_eva" \
