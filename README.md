
# Distributed multi-Agent Reinforcement Learning with One-hop Neighbors (DARL1N)

This is the code base  for implementing the DARL1N algorithm presented in the paper: [Distributed multi-Agent Reinforcement Learning with One-hop Neighbors](https://arxiv.org/abs/2202.09019) (DARL1N). This repository includes implementaions of four algorithms including DARL1N, [Evoluationary Population Curriculum](https://openreview.net/forum?id=SJxbHkrKDH) (EPC), [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1706.02275.pdf) (MADDPG) , and [Mean Field Actor Critic](https://arxiv.org/abs/1802.05438) (MFAC). The original implementaions of EPC, MFAC are contained in this [repository](https://github.com/qian18long/epciclr2020), and MADDPG is in this [repository](https://github.com/openai/maddpg).


## Dependancies:

- Known dependancies: python3 (3.6.9): numpy (1.19.2), gym (0.17.2), tensorflow (1.8.0), mpi4py (3.0.3), scipy (1.4.1), imageio (2.9.0), mpi4py (3.0.3); mpirun (Open MPI) (2.1.1), Ubuntu 18.04.4 LTS, ksh (sh (AT&T Research) 93u+ 2012-08-01).

- The DARL1N method is developed to run in a distributed computing system consisting of multiple computation nodes. In our paper, we use Amazon EC2 to build the computing system. Instructions of running our code on the Amazon EC2 is included in the directory `amazon_scripts`. You can also run our method in a single computation node, which will become multiprocessing instead of distributed computing.

- To run our code, first go to the root directory of this repository and install needed modules by `pip3 install -e .`



## Quick Start

### Training
- There are four directories `train_adversarial`, `train_grassland`, `train_ising`, `train_simple_spread`, including runnable scripts for the four methods in each environment.


### Evaluation
- There are four directories `evaluate_adversarial`, `evaluate_grassland`, `evaluate_ising`, `evaluate_simple_spread`, including runable scripts for the four methods in each environment. We provide the weights for each method in each environment with the small number of agents. You can directly run the evaluation scripts to evaluate and visualize trained agents with different methods in different environments. For Ising Model, the history of states are stored in the weight directory and needed to be plotted for visualization. Due to the file size limit of CMT system, we only provide weights for small scale settings.


## Training

### Command-line options


#### Environment options

- `--scenario`: defines which environment to be used (options: `ising`, `simple_spread`, `grassland`, `adversarial`)

- `--good-sight`: the good agent's visibility radius. (for MADDPG, MFAC and EPC, the value is set to `100`, which means the whole environment is observable, for DARL1N, this value corresponds to the neighbor distance and is set to other values smaller than the size of the environment, such as 0.2.)

- `--adv-sight`: the adversary agent's visibility radius, similar with the good sight.

- `--num-agents`: number of total agents.

- `--num-adversaries`: number of adversary agents.

- `--num-good`:number of good agents.

- `--num-food`: number of food (resources) in the scenario.

- `--max-episode-len` maximum length of each episode for the environment.

- `--ratio`: size of the environment space.

- `--num-episodes` total number of training iterations.

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment.
(default: `"maddpg"` (MADDPG and DARL1N); options: {`"att-maddpg"` (EPC), `"mean-field"` (MFAC)})

- `--adv-policy`: algorithm used for the adversary policies in the environment
algorithm used for the 'good' (non adversary) policies in the environment.
(default: `"maddpg"` (MADDPG and DARL1N); options: {`"att-maddpg"` (EPC), `"mean-field"` (MFAC)})


#### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--max-num-train`: maximum number of training iterations.

- `--seed`: set training seed for reproducibility. (For the EPC method, same seed may not lead to same result because environment processes share a common buffer and collect training data asynchronously and independently. The mini-batch sampled from the buffer with the same seed may differ due to different running speed of different processes.)


#### Checkpointing

- `--save-dir`: directory where intermediate training results and model will be saved.

- `--save-rate`: model is saved every time this number of training iterations has been completed.

- `--good-load-dir`: directory where training state and model of good agents are loaded from.

- `--adv-load-dir`: directory where training state and model of adversary agents are loaded from.

- `--adv-load-one-side`: load training state and model of adversary agents from the directory specified with `--adv-load-dir`.


#### Options for EPC

- `--n_cpu_per_agent`: cpu usage per agent (default: `1`)

- `--good-share-weights`: good agents share weights of the agents encoder within the model.

- `--adv-share-weights`: adversarial agents share weights of the agents encoder within the model.

- `--n-envs`: number of environments instances in parallelization.

- `--last-adv`: number of adversary agents in the last stage.

- `--last-good`: number of good agents in the last stage.

- `--good-load-dir1`: directory where training state and model of first hald of good agents are loaded from.

- `--good-load-dir2`: directory where training state and model of second hald of good agents are loaded from.

- `--timeout`: seconds to wait to get data from an empty Queue in multi-processing. If the get is not successful till the expiry of timeout seconds, an exception queue.

- `--restore`: restore training state and model from the specified load directories
(For the EPC method, you may also need to allow the system to use many processes by running the command `ulimit -n 20000` (or with a larger number) )

#### Options for DARL1N

- `--prosp-dist`: value to specify the potential neighbor, corresponding to \epsilon in the paper.
- `--num-learners`: number of learners in the distributed computing system.



## Evaluation

### Command line options:
Most options are same with training command line options. Here are other options.
- `--method`: method to use including `maddpg`, `mean_field`, `darl1n` (There is a separate script for `EPC` method).
- `--display`: displays to the screen the trained policy stored in the specified directories.


## Main files and directories desriptions:
- `.maddpg_o/experiments/train_normal.py`: train the schedules MADDPG or MFAC algorithm.

- `.maddpg_o/experiments/train_epc.py`: train the scheduled EPC algorithm.

- `.maddpg_o/experiments/train_darl1n.py`: train the scheduled DARL1N algorithm.

- `.maddpg_o/experiments/train_epc_select.py`: perform mutation and selection procedure for EPC.

- `.maddpg_o/experiments/evaluate_epc.py`: evaluation of EPC algorithm.

- `.maddpg_o/experiments/evaluate_normal.py`: evaluation of MADDPG, MFAC and EPC algorithms.

- `./maddpg_o/maddpg_local`: directory that contains helper functions for the training functions.

- `./mpe_local/multiagent/`: directory that contains code for different environments.

- `./amazon_scripts`: directory that contains scripts to coordinate the distributed computing system and run DARL1N algorithm on Amazon EC2.

- `./result`: directory that contains weights for each method in each environments.


## Demo
### Ising model (9 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/epc.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/9%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
### Ising model (16 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/mf.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/epc16_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/16%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
### Ising model (25 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/mf25.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/epc25_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/25%20agents/darl1n.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
 ### Ising model (64 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/maddpg.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/mean_field_local64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/epc64_ising_model.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/ising/64%20agents/darl1n64_ising_model.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
  ### Food collection (3 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/3%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/3%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Food collection (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/6%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/6%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Food collection (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/12%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/12%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Food collection (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/maddpg/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/mf/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/epc/24%20agents/208.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/simple_spread/darl1n/24%20agents/208.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
  ### Grassland (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Grassland (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Grassland (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Grassland (48 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/grassland/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (6 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/6%20agents/26.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/6%20agents/26.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
 
   ### Adversarial (12 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/12%20agents/31.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/12%20agents/31.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (24 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/24%20agents/36.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/24%20agents/36.gif" width="270" height="200" /></td>
  </tr>
 </table>
 
   ### Adversarial (48 agents)
<table>
  <tr>
    <td>MADDPG</td>
     <td>MFAC</td>
     <td>EPC</td>
    <td>DARL1N</td>
  </tr>
  <tr>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/maddpg/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/mf/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/epc/48%20agents/41.gif" width="270" height="200" /></td>
    <td><img src="https://github.com/BaoqianWang/IROS22_DARL1N/blob/master/demos/adversarial/darl1n/48%20agents/41.gif" width="270" height="200" /></td>
  </tr>
 </table>
