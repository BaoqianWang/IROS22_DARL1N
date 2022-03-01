import argparse
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pickle
from mpi4py import MPI
import random
import maddpg_o.maddpg_local.common.tf_util as U
from maddpg_o.maddpg_local.micro.maddpg_neighbor import MADDPGAgentTrainer
from maddpg_o.maddpg_local.micro.policy_target_policy import PolicyTrainer, PolicyTargetPolicyTrainer
import tensorflow.contrib.layers as layers
import json
import imageio
import joblib

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--eva-max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--max-num-train", type=int, default=2000, help="number of train")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="../trained_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=20, help="save model once every time this number of train are completed")
    parser.add_argument("--adv-load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--good-load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="../learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--num-good", type=int, default="0", help="num good")
    parser.add_argument("--num-landmarks", type=int, default="0", help="num landmarks")
    parser.add_argument("--num-agents", type=int, default="0", help="num agents")
    parser.add_argument("--num-learners", type=int, default="0", help="num learners")
    parser.add_argument("--last-good", type=int, default="2", help="num good agents from last stage")
    parser.add_argument("--last-adv", type=int, default="2", help="num adv agents from last stage")
    parser.add_argument("--good-max-num-neighbors", type=int, default="0", help="maximum number of  agents in neighbors area")
    parser.add_argument("--adv-max-num-neighbors", type=int, default="0", help="maximum number of  agents in neighbors area")
    parser.add_argument("--num-food", type=int, default="0", help="num food")
    parser.add_argument("--num-forests", type=int, default="0", help="num foresets")
    parser.add_argument("--prosp-dist", type=float, default="0.6", help="prospective neighbor distance")
    parser.add_argument("--adv-sight", type=float, default="1", help="neighbor distance")
    parser.add_argument("--good-sight", type=float, default="1", help="neighbor distance")
    parser.add_argument("--ratio", type=float, default="1", help="size of the map")
    parser.add_argument("--seed", type=int, default="1", help="seed for random number")
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--load-one-side", action="store_true", default=False)
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, evaluate=False): ###################
    import importlib
    if evaluate:
        from mpe_local.multiagent.environment import MultiAgentEnv
        module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        scenario = scenario_class(n_good=arglist.num_agents - arglist.num_adversaries, n_adv=arglist.num_adversaries, n_landmarks=arglist.num_landmarks, n_food=arglist.num_food, n_forests=arglist.num_forests,
                                      no_wheel=arglist.no_wheel, good_sight=arglist.good_sight, adv_sight=arglist.adv_sight, alpha=0, ratio = arglist.ratio, max_good_neighbor = arglist.good_max_num_neighbors, max_adv_neighbor = arglist.adv_max_num_neighbors)
    else:
        from mpe_local.multiagent.environment_neighbor import MultiAgentEnv
        module_name = "mpe_local.multiagent.scenarios.{}_neighbor".format(scenario_name)
        scenario_class = importlib.import_module(module_name).Scenario
        scenario = scenario_class(n_good=arglist.num_agents - arglist.num_adversaries, n_adv=arglist.num_adversaries, n_landmarks=arglist.num_landmarks, n_food=arglist.num_food, n_forests=arglist.num_forests,
                                      no_wheel=arglist.no_wheel, good_sight=arglist.good_sight, adv_sight=arglist.adv_sight, alpha=0, ratio = arglist.ratio, prosp=arglist.prosp_dist, max_good_neighbor = arglist.good_max_num_neighbors, max_adv_neighbor = arglist.adv_max_num_neighbors)

    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_agents, name, obs_shape_n, arglist, session):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            name+"agent_%d" % i, model, obs_shape_n, session, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def evaluate_policy(evaluate_env, trainers, num_episode, display = False):
    good_episode_rewards = [0.0]
    adv_episode_rewards = [0.0]
    step = 0
    episode = 0
    frames = []
    obs_n = evaluate_env.reset()
    while True:
        action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]

        new_obs_n, rew_n, done_n, next_info_n = evaluate_env.step(action_n)
        #print(rew_n)
        for i, rew in enumerate(rew_n):
            if i < arglist.num_adversaries:
                adv_episode_rewards[-1] += rew
            else:
                good_episode_rewards[-1] += rew

        step += 1
        done = all(done_n)
        terminal = (step > (arglist.eva_max_episode_len))

        obs_n = new_obs_n
        info_n = next_info_n

        if arglist.display:
            time.sleep(0.1)
            frames.append(evaluate_env.render('rgb_array')[0])
            print('The step is', step)
            if (terminal or done):
                imageio.mimsave('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode), frames, duration=0.15)
                frames=[]
                print('demo_num_agents_%d_%d.gif' %(arglist.num_agents, episode))

        if done or terminal:
            episode += 1
            if episode >= num_episode:
                break
            #print(good_episode_rewards[-1])
            #print(adv_episode_rewards[-1])
            good_episode_rewards.append(0)

            adv_episode_rewards.append(0)
            obs_n = evaluate_env.reset()
            step = 0

    return np.mean(good_episode_rewards), np.mean(adv_episode_rewards)


def interact_with_environments(env, trainers, node_id, steps):
    act_d = env.action_space[0].n
    for k in range(steps):
        obs_pot, neighbor = env.reset(node_id) # Neighbor does not include agent itself

        action_n = [np.zeros((act_d))] * env.n # Actions for transition

        action_neighbor = [np.zeros((act_d))] * arglist.good_max_num_neighbors #The neighbors include the agent itself
        target_action_neighbor = [np.zeros((act_d))] * arglist.good_max_num_neighbors

        self_action = trainers[node_id].action(obs_pot[node_id])

        action_n[node_id] = self_action
        action_neighbor[0] = self_action

        valid_neighbor = 1
        for i, obs in enumerate(obs_pot):
            if i == node_id: continue
            if len(obs) !=0 :
                #print(obs)
                other_action = trainers[i].action(obs)
                action_n[i] = other_action
                if neighbor and i in neighbor and valid_neighbor < arglist.good_max_num_neighbors:
                    action_neighbor[valid_neighbor] = other_action
                    valid_neighbor += 1
        #print(action_n)
        new_obs_neighbor, rew, done_n, next_info_n = env.step(action_n) # Interaction within the neighbor area

        valid_neighbor = 1
        target_action_neighbor[0]=trainers[node_id].target_action(new_obs_neighbor[node_id])

        for k, next in enumerate(new_obs_neighbor):
            if k == node_id: continue
            if len(next) != 0 and valid_neighbor < arglist.good_max_num_neighbors:
                target_action_neighbor[valid_neighbor] = trainers[k].target_action(next)
                valid_neighbor += 1

        info_n = 0.1
        trainers[node_id].experience(obs_pot[node_id], action_neighbor, new_obs_neighbor[node_id], target_action_neighbor, rew)

    return


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_weights(trainers, index):
    weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %index)
    touch_path(weight_file_name)
    weight_dict = trainers[index].get_all_weights()
    joblib.dump(weight_dict, weight_file_name)


def load_weights(trainers, index):
    # with open(arglist.save_dir + 'agent%d.weights' %i,'rb') as f:
    #     weight_dict=pickle.load(f)

    # Attention here
    if index >= arglist.num_adversaries:
        weight_dict = joblib.load(os.path.join(arglist.good_load_dir, 'agent%d.weights' %((index-arglist.num_adversaries)%arglist.last_good + arglist.last_adv)))
        trainers[index].set_all_weights(weight_dict)
    else:
        weight_dict = joblib.load(os.path.join(arglist.adv_load_dir, 'agent%d.weights' %(index%arglist.last_adv)))
        trainers[index].set_all_weights(weight_dict)


# def save_weights(trainers):
#     for i in range(len(trainers)):
#         weight_file_name = os.path.join(arglist.save_dir, 'agent%d.weights' %i)
#         touch_path(weight_file_name)
#         weight_dict = trainers[i].get_weigths()
#         joblib.dump(weight_dict, weight_file_name)
        # with open(weight_file_name, 'w+') as fp:
        #     pickle.dump(weight_dict, fp)

# def load_weights(trainers):
#     for i in range(len(trainers)):
#         # with open(arglist.save_dir + 'agent%d.weights' %i,'rb') as f:
#         #     weight_dict=pickle.load(f)
#         weight_dict = joblib.load(os.path.join(arglist.load_dir, 'agent%d.weights' %(i%arglist.last_stage_num)))
#         trainers[i].set_weigths(weight_dict)


if __name__== "__main__":
    # MPI initialization.
    comm = MPI.COMM_WORLD
    num_node = comm.Get_size()
    node_id = comm.Get_rank()
    node_name = MPI.Get_processor_name()

    with tf.Session() as session:
        #Parse the parameters
        arglist = parse_args()
        seed = arglist.seed
        gamma = arglist.gamma
        num_agents = arglist.num_agents
        num_learners = arglist.num_learners # In two sided applications, we only train one side.
        assert num_node == num_learners + 1
        np.random.seed(seed)
        tf.set_random_seed(seed)
        random.seed(seed)
        CENTRAL_CONTROLLER = arglist.num_adversaries
        LEARNERS = [i+CENTRAL_CONTROLLER for i in range(1, 1+num_learners)]
        env = make_env(arglist.scenario, arglist, evaluate= False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        node_id += arglist.num_adversaries

        if (node_id == CENTRAL_CONTROLLER):
            trainers = []
            # The central controller only needs policy parameters to execute the policy for evaluation
            for i in range(num_agents):
                trainers.append(PolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
        else:
            trainers = []
            # Trainer needs the MADDPG trainer for its own agent, while only needs policy and target policy for good agents, and policy for adversary agents
            for i in range(num_agents):
                if node_id - 1 == i:
                    trainers.append(MADDPGAgentTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
                elif i >= arglist.num_adversaries: # Good agents
                    trainers.append(PolicyTargetPolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))
                else: # Adversary agents
                    trainers.append(PolicyTargetPolicyTrainer("actor" + "agent_%d" % i, mlp_model, obs_shape_n, session, env.action_space, i, arglist, local_q_func=False))


        U.initialize()
        final_good_rewards = []
        final_adv_rewards = []
        final_rewards = []
        train_time = []
        global_train_time = []
        ground_global_time = time.time()
        train_step = 0
        iter_step = 0
        num_train = 0

        if (node_id == CENTRAL_CONTROLLER):
            train_start_time = time.time()
            print('Computation scheme: ', 'DARL1N')
            print('Scenario: ', arglist.scenario)
            print('Number of agents: ', num_agents)
            touch_path(arglist.save_dir)
            # if arglist.load_dir == "":
            #     arglist.load_dir = arglist.save_dir
            print('Good load dir is', arglist.good_load_dir)
            print('Adv load dir is', arglist.adv_load_dir)
            evaluate_env = make_env(arglist.scenario, arglist, evaluate= True)

        if arglist.load_one_side:
            print('Loading one side state...')
            # Load adversary agents weights
            one_side_weights = None
            if node_id > CENTRAL_CONTROLLER:
                load_weights(trainers, node_id - 1 - arglist.num_adversaries)
                one_side_weights = trainers[node_id - 1 - arglist.num_adversaries].get_weigths()
            one_side_weights = comm.gather(one_side_weights, root = 0)
            one_side_weights = comm.bcast(one_side_weights, root = 0)
            if node_id > CENTRAL_CONTROLLER:
                for i, agent in enumerate(trainers):
                    if i < arglist.num_adversaries:
                        agent.set_weigths(one_side_weights[i+1])


        if arglist.restore:
            # Load good agents weights from the last stage
            print('Loading previous state...')
            touch_path(arglist.good_load_dir)
            touch_path(arglist.adv_load_dir)
            weights = None
            if node_id > CENTRAL_CONTROLLER:
                # load weights for each good agent in each learner
                load_weights(trainers, node_id - 1)
                weights = trainers[node_id - 1].get_weigths()
            # Gather policy and target policy
            weights = comm.gather(weights, root = 0)
            # Broadcast policy and target policy parameters of all agents to each agent
            weights = comm.bcast(weights,root=0)

            if node_id > CENTRAL_CONTROLLER:
            # For each learner, set the policy and target policy for each agent
                for i, agent in enumerate(trainers):
                    if i >= arglist.num_adversaries:
                        agent.set_weigths(weights[i+1-arglist.num_adversaries])

        comm.Barrier()
        print('Start training...')
        start_time = time.time()
        while True:
            comm.Barrier()
            if num_train > 0:
                start_master_weights=time.time()
                weights=comm.bcast(weights,root=0)
                end_master_weights=time.time()

            if (node_id in LEARNERS):
                # Receive parameters
                if num_train == 0:
                    env_time1 = time.time()
                    interact_with_environments(env, trainers, node_id-1, 5 * arglist.batch_size)
                    env_time2 = time.time()
                    print('Env interaction time', env_time2 - env_time1)
                else:
                    for i, agent in enumerate(trainers):
                        if i >= arglist.num_adversaries:
                            agent.set_weigths(weights[i+1-arglist.num_adversaries])
                    interact_with_environments(env, trainers, node_id-1, 4 * arglist.eva_max_episode_len)

                loss = trainers[node_id-1].update(trainers)
                weights = trainers[node_id-1].get_weigths()

            if (node_id == CENTRAL_CONTROLLER):
                weights = None

            weights = comm.gather(weights, root = 0)

            if (node_id in LEARNERS):
                num_train += 1
                if num_train > arglist.max_num_train:
                    save_weights(trainers, node_id - 1)
                    break

            if(node_id == CENTRAL_CONTROLLER):
                if(num_train % arglist.save_rate == 0):
                    for i in range(num_agents):
                        if i < arglist.num_adversaries:
                            trainers[i].set_weigths(one_side_weights[i+1])
                        else:
                            trainers[i].set_weigths(weights[i+1-arglist.num_adversaries])

                    end_train_time = time.time()
                    #U.save_state(arglist.save_dir, saver=saver)
                    good_reward, adv_reward = evaluate_policy(evaluate_env, trainers, 10, display = False)
                    final_good_rewards.append(good_reward)
                    final_adv_rewards.append(adv_reward)
                    train_time.append(end_train_time - start_time)
                    print('Num of training iteration:', num_train, 'Good Reward:', good_reward, 'Adv Reward:', adv_reward, 'Training time:', round(end_train_time - start_time, 3), 'Global training time:', round(end_train_time- ground_global_time, 3))
                    global_train_time.append(round(end_train_time - ground_global_time, 3))
                    start_time = time.time()
                num_train += 1
                if num_train > arglist.max_num_train:
                    #save_weights(trainers)
                    good_rew_file_name = arglist.save_dir + 'good_agent.pkl'
                    with open(good_rew_file_name, 'wb') as fp:
                        pickle.dump(final_good_rewards, fp)

                    adv_rew_file_name = arglist.save_dir  + 'adv_agent.pkl'
                    with open(adv_rew_file_name, 'wb') as fp:
                        pickle.dump(final_adv_rewards, fp)

                    time_file_name = arglist.save_dir + 'train_time.pkl'
                    with open(time_file_name, 'wb') as fp:
                        pickle.dump(train_time, fp)

                    global_time_file = arglist.save_dir + 'global_time.pkl'
                    with open(global_time_file, 'wb') as fp:
                        pickle.dump(global_train_time, fp)

                    train_end_time = time.time()
                    print('The total training time:', train_end_time - train_start_time)
                    print('Average train time', np.mean(train_time))
                    break
