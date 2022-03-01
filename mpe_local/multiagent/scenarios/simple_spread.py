import numpy as np
from mpe_local.multiagent.core import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os

class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, good_sight, adv_sight, no_wheel, ratio, max_good_neighbor, max_adv_neighbor):
        self.n_good = n_good
        self.n_adv = n_adv
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.num_agents = n_adv + n_good
        self.alpha = alpha
        self.good_neigh_dist = good_sight
        self.adv_neigh_dist = adv_sight
        self.ratio = ratio
        self.size = ratio
        self.no_wheel = no_wheel
        self.max_good_neighbor = max_good_neighbor
        self.max_adv_neighbor = max_adv_neighbor


    def make_world(self):
        world = World()
        world.dim_c = 2
        world.size = self.ratio
        world.good_neigh_dist = self.good_neigh_dist
        world.adv_neigh_dist = self.adv_neigh_dist
        world.max_good_neighbor = self.max_good_neighbor
        world.max_adv_neighbor = self.max_adv_neighbor
        num_agents = self.n_good
        self.num_agents = num_agents
        num_landmarks = self.n_food
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.live = 1
            agent.adversary = False
            agent.max_speed = 3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        world.food = world.landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])#np.random.uniform(0, 1, 3)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])#world.agents[i].color
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.ratio*1, +self.ratio*1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-self.ratio*1, +self.ratio*1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for agent in world.agents:
            dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) for landmark in world.landmarks]


    def done(self, agent, world):
        return 0


    def info(self, agent, world):
        return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)
        for food in world.landmarks:
            if self.is_collision(food, agent):
                rew += 1
        for a in world.agents:
            if a == agent: continue
            if self.is_collision(a, agent):
                rew -= 1
        return rew

    def observation(self, agent, world):
        # current agent
        dist = []
        for i, landmark in enumerate(world.landmarks):
            dist.append((i, np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
        dist = sorted(dist, key = lambda t: t[1])
        entity_pos = []
        for i, land_dist in dist:
            entity_pos.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
            entity_pos.append([0])

        for j in range(self.max_good_neighbor - self.num_agents):
            entity_pos.append([0, 0])
            entity_pos.append([0])

        other_pos = [[0, 0] for i in range(self.max_good_neighbor-1)]
        other_live = [[0] for i in range(self.max_good_neighbor-1)]
        other_vel = [[0, 0] for i in range(self.max_good_neighbor-1)]
        num_neighbor = 0
        for i, other in enumerate(world.agents):
            if other is agent: continue
            distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if distance <= self.good_neigh_dist and num_neighbor < self.max_good_neighbor-1:
                other_vel[num_neighbor] = other.state.p_vel
                other_pos[num_neighbor] = other.state.p_pos- agent.state.p_pos
                other_live[num_neighbor] = [1]
                num_neighbor += 1

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([1])] + entity_pos + other_pos + other_vel + other_live)
