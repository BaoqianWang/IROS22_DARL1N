#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=0
l=0


while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < nodeIPaddress




sleep 1
echo "Train DARL1N Adversarial "

# 6 agents
host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=4;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done

mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
python3 -m maddpg_o.experiments.train_darl1n \
    --scenario=adversarial \
    --good-sight=0.2 \
    --adv-sight=100 \
    --num-agents=6 \
    --num-learners=3 \
    --num-adversaries=3 \
    --num-food=3 \
    --num-landmark=3\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/adversarial/darl1n/6agents/6agents_1/" \
    --adv-load-dir="../result/adversarial/baseline_maddpg/6agents/6agents_1/" \
    --save-rate=30 \
    --prosp-dist=0.1 \
    --eva-max-episode-len=25 \
    --good-max-num-neighbors=6 \
    --adv-max-num-neighbors=6 \
    --max-num-train=2000\
    --eva-max-episode-len=25 \
    --max-episode-len=25 \
    --ratio=1.5 \
    --seed=16\
    --load-one-side \


# 12 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=7;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done
#
# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3 -m maddpg_o.experiments.train_darl1n \
#     --scenario=adversarial \
#     --good-sight=0.25 \
#     --adv-sight=100 \
#     --num-agents=12 \
#     --num-learners=6 \
#     --num-adversaries=6 \
#     --num-learners=6 \
#     --num-food=6 \
#     --num-landmark=6\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/darl1n/12agents/12agents_1/" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/12agents/12agents_1/" \
#     --save-rate=30 \
#     --prosp-dist=0.15 \
#     --good-max-num-neighbors=12 \
#     --adv-max-num-neighbors=12 \
#     --max-num-train=3000\
#     --eva-max-episode-len=30 \
#     --max-episode-len=30 \
#     --ratio=2 \
#     --seed=16\
#     --load-one-side \


# 24 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=13;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done
#
# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3 -m maddpg_o.experiments.train_darl1n \
#     --scenario=adversarial \
#     --good-sight=0.3 \
#     --adv-sight=100 \
#     --num-agents=24 \
#     --num-learners=12 \
#     --num-adversaries=12 \
#     --num-learners=12 \
#     --num-food=12 \
#     --num-landmark=12\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/darl1n/24agents/24agents_1/" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/24agents/24agents_1/" \
#     --save-rate=30 \
#     --prosp-dist=0.2 \
#     --eva-max-episode-len=35 \
#     --max-episode-len=35 \
#     --good-max-num-neighbors=24 \
#     --adv-max-num-neighbors=24 \
#     --max-num-train=3000\
#     --ratio=2.5 \
#     --seed=16\
#     --load-one-side \

# 48 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=25;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done

# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3 -m maddpg_o.experiments.train_darl1n \
#     --scenario=adversarial \
#     --good-sight=0.35 \
#     --adv-sight=100 \
#     --num-agents=48 \
#     --num-learners=24 \
#     --num-adversaries=24 \
#     --num-food=24 \
#     --num-landmark=24\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/adversarial/darl1n/48agents/48agents_1/" \
#     --adv-load-dir="../result/adversarial/baseline_maddpg/48agents/48agents_1/" \
#     --save-rate=30 \
#     --prosp-dist=0.25 \
#     --eva-max-episode-len=40 \
#     --max-episode-len=40 \
#     --good-max-num-neighbors=48 \
#     --adv-max-num-neighbors=48 \
#     --max-num-train=2000\
#     --eva-max-episode-len=40 \
#     --max-episode-len=40 \
#     --ratio=3 \
#     --seed=16\
#     --load-one-side \
