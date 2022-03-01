import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import layers
import math
from maddpg_o.maddpg_local.micro.maddpg import MADDPGAgentMicroSharedTrainer
from maddpg_o.maddpg_local.micro.policy_target_policy import PolicyTrainer
import maddpg_o.maddpg_local.common.tf_util as U
from .train_helper.model_v3_test3 import mlp_model_agent_p, mlp_model_agent_p_ising, mlp_model_agent_q_ising, mlp_model_adv_p,  mlp_model_agent_q, mlp_model_adv_q, mlp_model, mean_field_adv_q_model, mean_field_agent_q_model
import argparse
import time
import re
# import ray
from functools import partial
from maddpg_o.experiments.train_helper.union_replay_buffer import UnionReplayBuffer
import numpy as np
import imageio
import queue
import joblib
import tempfile
import random
import gc
import imageio
import matplotlib.pyplot as plt
import pickle

FLAGS = None
# import multiagent
# print(multiagent.__file__)
# print(Scenario)

# ray.init()

# load model
# N_GOOD, N_ADV, N_LAND should align with the environment
N_GOOD = None
N_ADV = None
# N_LAND = num_landmarks+num_food+num_forests
N_LANDMARKS = None
N_FOOD = None
N_FORESTS = None
N_LAND = None

ID_MAPPING = None

INIT_WEIGHTS = None
GOOD_SHARE_WEIGHTS = False
ADV_SHARE_WEIGHTS = False
SHARE_WEIGHTS = None
CACHED_WEIGHTS = {}

WEIGHT_STACK = False

GRAPHS = []
SESSIONS = []
TRAINERS = []

CLUSTER = None
SERVERS = None

PERTURBATION = None
NEW_IDS = []


def register_environment(n_good, n_adv, n_landmarks, n_food, n_forests, init_weights, id_mapping=None):
    global N_GOOD, N_ADV, N_LAND, N_LANDMARKS, N_FOOD, N_FORESTS, ID_MAPPING, INIT_WEIGHTS
    N_GOOD = n_good
    N_ADV = n_adv
    N_LANDMARKS = n_landmarks
    N_FOOD = n_food
    N_FORESTS = n_forests
    N_LAND = N_LANDMARKS + N_FOOD + N_FORESTS
    INIT_WEIGHTS = init_weights
    ID_MAPPING = id_mapping
    # print("SHARE_WEIGHTS", SHARE_WEIGHTS)


def make_env(scenario_name, arglist, benchmark=False):
    import importlib
    from mpe_local.multiagent.environment import MultiAgentEnv
    module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
    scenario_class = importlib.import_module(module_name).Scenario
    # load scenario from script
    # print(Scenario.__module__.__file__)
    scenario = scenario_class(n_good=N_GOOD, n_adv=N_ADV, n_landmarks=N_LANDMARKS, n_food=N_FOOD, n_forests=N_FORESTS,
                              no_wheel=FLAGS.no_wheel, good_sight=FLAGS.good_sight, alpha=FLAGS.alpha, adv_sight=FLAGS.adv_sight, ratio=FLAGS.ratio, max_good_neighbor = N_GOOD + N_ADV, max_adv_neighbor = N_GOOD + N_ADV)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, done_callback=scenario.done, info_callback=scenario.info,
                        export_episode=FLAGS.save_gif_data)
    return env


def make_session(graph, num_cpu):
    # print("num_cpu:", num_cpu)
    tf_config = tf.ConfigProto(
        # device_count={"CPU": num_cpu},
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    return tf.Session(graph=graph, config=tf_config)
    # return tf.Session(target=server.target, graph=graph, config=tf_config)


def get_trainer(side, i, scope, env, obs_shape_n):
    trainer = MADDPGAgentMicroSharedTrainer
    policy = FLAGS.adv_policy if side == "adv" else FLAGS.good_policy
    share_weights = FLAGS.adv_share_weights if side == "adv" else FLAGS.good_share_weights
    if policy == "att-maddpg":

        if FLAGS.scenario == 'ising':
            model_p = partial(mlp_model_agent_p_ising, n_good=N_GOOD, n_adv=N_ADV,
                              n_land=N_LAND, index=i, share_weights=share_weights)
            model_q = partial(mlp_model_agent_q_ising, n_good=N_GOOD, n_adv=N_ADV,
                              n_land=N_LAND, index=i, share_weights=share_weights)
        else:
            model_p = partial(mlp_model_adv_p if side == "adv" else mlp_model_agent_p, n_good=N_GOOD, n_adv=N_ADV,
                              n_land=N_LAND, index=i, share_weights=share_weights)
            model_q = partial(mlp_model_adv_q if side == "adv" else mlp_model_agent_q, n_good=N_GOOD, n_adv=N_ADV,
                              n_land=N_LAND, index=i, share_weights=share_weights)

    elif policy == "maddpg":
        model_p = mlp_model
        model_q = mlp_model
    elif policy == "mean_field":
        model_p = mlp_model
        model_q = partial(mean_field_adv_q_model if side == "adv" else mean_field_agent_q_model, n_good=N_GOOD,
                          n_adv=N_ADV, n_land=N_LAND, index=i)
    else:
        raise NotImplementedError
    # print(obs_shape_n)
    num_units = (FLAGS.adv_num_units if side == "adv" else FLAGS.good_num_units) or FLAGS.num_units
    return trainer(scope, model_p, model_q, obs_shape_n, env.action_space, i, FLAGS, num_units, local_q_func=False)


def get_adv_trainer(i, scope, env, obs_shape_n):
    return get_trainer("adv", i, scope, env, obs_shape_n)

def get_one_side_trainer(i, scope, env, obs_shape_n):
    trainer = PolicyTrainer
    return trainer(scope, mlp_model, obs_shape_n, None, env.action_space, i, FLAGS, local_q_func=False)

def get_good_trainer(i, scope, env, obs_shape_n):
    return get_trainer("good", i, scope, env, obs_shape_n)


def show_size():
    s = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        tot = 1
        for dim in shape:
            tot *= dim
        s += tot


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def load_weights(load_path):
    import joblib
    global CACHED_WEIGHTS

    CACHED_WEIGHTS.update(joblib.load(load_path))


def clean(d):
    rd = {}
    for k, v in d.items():
        # if v.shape[0] == 456 or v.shape[0] == 1552:
        #     print(k, v.shape)
        if type(k) == tuple:
            rd[k[0]] = v
        else:
            rd[k] = v
    return rd


def load_all_weights(load_dir, n):
    global CACHED_WEIGHTS
    CACHED_WEIGHTS = {}
    for i in range(n):
        # print(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
        load_weights(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
    # print(CACHED_WEIGHTS)
    CACHED_WEIGHTS = clean(CACHED_WEIGHTS)


def parse_args(add_extra_flags=None):
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="grassland",
                        help="name of the scenario script")
    parser.add_argument("--map-size", type=str, default="normal")
    parser.add_argument("--good-sight", type=float, default=100)
    parser.add_argument("--adv-sight", type=float, default=100)
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--show-attention", action="store_true", default=False)
    parser.add_argument("--max-episode-len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int,
                        default=200000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int,
                        default=2, help="number of adversaries")
    parser.add_argument("--num-good", type=int,
                        default=2, help="number of good")
    parser.add_argument("--num-agents", type=int,
                        default=2, help="number of agents")
    parser.add_argument("--num-food", type=int,
                        default=4, help="number of food")
    parser.add_argument("--good-policy", type=str,
                        default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str,
                        default="maddpg", help="policy of adversaries")
    parser.add_argument("--good-load-one-side", action="store_true", default=False)
    parser.add_argument("--adv-load-one-side", action="store_true", default=False)
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--good-num-units", type=int)
    parser.add_argument("--adv-num-units", type=int)
    parser.add_argument("--n-cpu-per-agent", type=int, default=1)
    parser.add_argument("--good-share-weights", action="store_true", default=False)
    parser.add_argument("--adv-share-weights", action="store_true", default=False)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--good-save-dir", type=str, default="./test/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--adv-save-dir", type=str, default="./test/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--train-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--checkpoint-rate", type=int, default=0)
    parser.add_argument("--load-dir", type=str, default="./test/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-gif-data", action="store_true", default=False)
    parser.add_argument("--render-gif", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000,
                        help="number of iterations run for benchmarking")

    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=1)
    parser.add_argument("--save-summary", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=0.02)

    if add_extra_flags is not None:
        parser = add_extra_flags(parser)

    return parser.parse_args()


with tf.Session() as session:
    #global FLAGS
    FLAGS = parse_args()
    arglist = FLAGS
    register_environment(n_good=FLAGS.num_good, n_adv=FLAGS.num_adversaries, n_landmarks=0, n_food=FLAGS.num_food,
                             n_forests=0, init_weights=None, id_mapping=None)

    tf.set_random_seed(30)
    random.seed(30)
    np.random.seed(30)

    env = make_env(FLAGS.scenario, FLAGS, FLAGS.benchmark)
    n = env.n
    obs_shape_n = [env.observation_space[i].shape for i in range(n)]
    action_shape_n = [env.action_space[i].n for i in range(n)]
    num_adversaries = FLAGS.num_adversaries

    trainers = []
    trainers_name = []
    print('start evaluation')
    for i in range(num_adversaries):
        trainers.append(get_one_side_trainer(i=i, scope = "{}".format(i), env = env, obs_shape_n = obs_shape_n))
        trainers_name.append("adv{}".format(i))
        print('Initialize', i)

    for i in range(num_adversaries, n):
        trainers.append(get_good_trainer(i=i, scope = "{}".format(i), env = env, obs_shape_n = obs_shape_n))
        trainers_name.append("good{}".format(i))
        print('Initialize', i)

    U.initialize()
    print('Initialization done')

    all_weights = dict()
    for i in range(num_adversaries, n):
        path = FLAGS.good_save_dir + 'agent%d.trainable-weights' %i
        weights = joblib.load(path)

        all_weights.update(weights)

    restores_op = []
    # for i, name in enumerate(trainers_name):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # path = FLAGS.save_dir + 'agent%d.weights' %i
    # weights = joblib.load(path)
    for variable in variables:
        if variable.name in all_weights:
            #print('restore')
            restores_op.append(variable.assign(all_weights[variable.name]))

    session.run(restores_op)

    for i in range(num_adversaries):
        trainers[i].session = session
        weight_dict = joblib.load(os.path.join(arglist.adv_save_dir, 'agent%d.weights' %(i)))
        trainers[i].set_all_weights(weight_dict)


    tf.set_random_seed(30)
    random.seed(30)
    np.random.seed(30)
    obs_n = env.reset()
    trans = 0
    episode_step = 0
    episode_reward = [0]
    frames = []
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    initial = []
    action_history = []
    if arglist.scenario == 'ising':
        for agent in env.world.agents:
            initial.append(agent.state.spin)
    #print(initial)
    while True:
        trans += 1
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
        action_history.append(action_n)
        #print(action_n)
        # environment step
        new_obs_n, rew_n, done_n, next_info_n = env.step(action_n)
        for i, rew in enumerate(rew_n):
            if i < arglist.num_adversaries:
                adv_episode_rewards[-1] += rew
            else:
                good_episode_rewards[-1] += rew

        #print('Next Observation', new_obs_n, 'Observation', obs_n, 'Action', action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step > FLAGS.max_episode_len)
        obs_n = new_obs_n
        info_n = next_info_n


        if arglist.scenario != 'ising':
            time.sleep(0.1)
            frames.append(env.render('rgb_array')[0])

        touch_path(arglist.good_save_dir)
        if done or terminal:
            #print(good_episode_rewards)
            if arglist.scenario == 'ising':
                action_file = arglist.good_save_dir + 'history_action%d.pkl' %trans
                with open(action_file, 'wb') as fp:
                    pickle.dump(action_history, fp)
                initial_file = arglist.good_save_dir + 'initial%d.pkl' %trans
                with open(initial_file, 'wb') as fp:
                    pickle.dump(initial, fp)
                initial =[]
                action_history = []
                obs_n = env.reset()
                episode_step = 0
                good_episode_rewards.append(0)
                adv_episode_rewards.append(0)
                if arglist.scenario == 'ising':
                    for agent in env.world.agents:
                        initial.append(agent.state.spin)

            else:
                gif_path = '../visualize/' + arglist.scenario + '/real_epc_vs_darl1n_adv_sight' + '/%dagents/gifs/' %n
                touch_path(gif_path)
                imageio.mimsave(gif_path + '%d.gif' %trans, frames, duration=0.15)
                plt.imshow(frames[-1])
                plt.xticks([]),plt.yticks([])
                plt.savefig(gif_path + '%d.png' %trans, transparent=True)
                frames=[]
                obs_n = env.reset()
                episode_step = 0
                good_episode_rewards.append(0)
                adv_episode_rewards.append(0)

        if trans >= 10*FLAGS.max_episode_len:
            print('Good reward: ', np.mean(good_episode_rewards), 'Adv reward: ', np.mean(adv_episode_rewards))
            print('Good reward', good_episode_rewards, 'Adv reward: ', adv_episode_rewards)
            break
