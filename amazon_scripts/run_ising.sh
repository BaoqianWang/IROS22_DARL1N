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
echo "Train DARL1N Ising"

# 9 agents
host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=10;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done

mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
python3 -m maddpg_o.experiments.train_darl1n \
    --scenario=ising \
    --good-sight=100 \
    --adv-sight=100 \
    --last-good=0 \
    --last-adv=0 \
    --num-agents=9 \
    --num-adversaries=0 \
    --num-learners=9 \
    --num-food=0 \
    --num-landmark=0\
    --good-policy=maddpg \
    --adv-policy=maddpg \
    --save-dir="../result/ising/darl1n/9agaents/9agents_1/" \
    --save-rate=10 \
    --prosp-dist=0.3 \
    --max-num-train=200 \
    --good-max-num-neighbors=5 \
    --adv-max-num-neighbors=5 \
    --eva-max-episode-len=25 \
    --max-episode-len=25 \
    --batch-size=32 \
    --seed=16 \

# 16 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=17;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done
#
# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --good-sight=100 \
#     --adv-sight=100 \
#     --last-good=0 \
#     --last-adv=0 \
#     --num-agents=16 \
#     --num-adversaries=0 \
#     --num-learners=16 \
#     --num-food=0 \
#     --num-landmark=0\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/16agents/16agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=600 \
#     --good-max-num-neighbors=5 \
#     --adv-max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \




# 25 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=26;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done
#
# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --good-sight=100 \
#     --adv-sight=100 \
#     --last-good=0 \
#     --last-adv=0 \
#     --num-agents=25 \
#     --num-adversaries=0 \
#     --num-learners=25 \
#     --num-food=0 \
#     --num-landmark=0\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/25agents/25agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=600 \
#     --good-max-num-neighbors=5 \
#     --adv-max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \


# 64 agents
# host_name_uncoded="${ARRAY[1]}"
# for((i=2;i<=65;i++))
# do
#   host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
# done
#
# mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
# python3  -m maddpg_o.experiments.train_darl1n \
#     --scenario=ising \
#     --good-sight=100 \
#     --adv-sight=100 \
#     --last-good=0 \
#     --last-adv=0 \
#     --num-agents=64 \
#     --num-adversaries=0 \
#     --num-learners=64 \
#     --num-food=0 \
#     --num-landmark=0\
#     --good-policy=maddpg \
#     --adv-policy=maddpg \
#     --save-dir="../result/ising/darl1n/64agents/64agents_1/" \
#     --save-rate=10 \
#     --prosp-dist=0.3 \
#     --max-num-train=250 \
#     --good-max-num-neighbors=5 \
#     --adv-max-num-neighbors=5 \
#     --eva-max-episode-len=25 \
#     --max-episode-len=25 \
#     --batch-size=32 \
#     --seed=16 \
