import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from mpe_local.multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.obs_dim_n = []
        for i, agent in enumerate(self.agents):
            if hasattr(world, 'ising'):
                if i < self.world.max_good_neighbor:
                    self.action_space.append(spaces.Discrete(self.world.dim_spin))
                self.observation_space.append(spaces.MultiBinary(5 * self.world.agent_view_sight))
                #self.obs_dim_n.append(4)
            else:
                if i < self.world.max_good_neighbor:
                    self.action_space.append(spaces.Discrete(world.dim_p * 2 + 1))
                self.reset_callback(self.world, i)
                obs_dim = len(observation_callback(i, self.world))
                self.obs_dim_n.append(obs_dim)
                self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()


    def step(self, action_n):
        #obs_n = [[]] *self.n
        # action_n is the full set of all agents
        agent = self.world.agents[self.agent_id]

        for i in self.action_agents:
            self._set_action(action_n[i], self.agents[i])

        self.world.step(self.action_agents)

        # obs_n = [[np.zeros((self.obs_dim_n[i]))] for i in range(self.n)]
        obs_n = [[] for i in range(self.n)]
        obs_n[self.agent_id] = self._get_obs(self.agent_id)
        neighbors = []
        if not hasattr(self.world, 'ising'):

            for i, other in enumerate(self.world.agents):
                if other is agent: continue
                distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
                if distance < self.world.good_neigh_dist:
                    obs_n[i] = self._get_obs(i)
                    neighbors.append(i)
            agent.neighbors = neighbors

        else:
            self_agent_neg = list(np.where(self.world.agents[self.agent_id].spin_mask == 1)[0])
            for other in self_agent_neg:
                obs_n[other] = self._get_obs(other)

        reward = self._get_reward(self.agent_id)

        return obs_n, reward, None, None


    def reset(self, agent_id, step = 0):
        # Set the central agent
        self.agent_id = agent_id
        # reset world
        self.action_agents, self.neighbor = self.reset_callback(self.world, agent_id, step)
        self._reset_render()
        obs_n = [[] for i in range(self.n)]
        for i in self.action_agents:
            obs_n[i] = self._get_obs(i)

        return obs_n, self.neighbor


    # get info used for benchmarking
    def _get_info(self, i, agent):
        neighbors = []
        for j, other in enumerate(self.world.policy_agents):
            if other==agent: continue
            dist_ij = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if(dist_ij<=self.world.neigh_dist):
                neighbors.append(j)

        return neighbors

    # get observation for a particular agent
    def _get_obs(self, i):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(i, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, i):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(i, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p)
            agent.action.c = np.zeros(self.world.dim_c)
            # physical action
            agent.action.u[0] += action[1] - action[2]
            agent.action.u[1] += action[3] - action[4]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

        else:
            agent.action.a = 0 if action[0] <= 0.5 else 1

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
