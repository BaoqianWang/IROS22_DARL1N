import numpy as np
from mpe_local.multiagent.core_neighbor import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, good_sight, adv_sight, no_wheel, ratio, prosp, max_good_neighbor, max_adv_neighbor):
        self.n_good = n_good
        self.n_landmarks = n_food
        self.n_food = n_food
        self.n_adv = n_adv
        self.n_forests = n_forests
        self.alpha = alpha
        self.ratio = ratio
        self.good_neigh_dist = good_sight
        self.adv_neigh_dist = adv_sight
        self.no_wheel = no_wheel
        self.neigh_dist = good_sight
        self.prosp_dist = prosp
        self.max_good_neighbor = max_good_neighbor
        self.max_adv_neighbor = max_adv_neighbor


    def make_world(self):
        # np.random.seed(24)
        world = World()
        world.dim_c = 2
        world.size = self.ratio
        self.num_agents = self.n_good + self.n_adv
        num_landmarks = self.n_food
        num_agents = self.num_agents
        world.collaborative = True
        world.good_neigh_dist = self.good_neigh_dist
        world.adv_neigh_dist = self.adv_neigh_dist
        world.max_good_neighbor = self.max_good_neighbor
        world.max_adv_neighbor = self.max_adv_neighbor
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.color = np.array([0.15, 0.65, 0.15])
            agent.state.c = np.zeros(self.num_agents)
            agent.max_speed = 3
            agent.live = 1

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.occupied = -1
            landmark.color  = np.array([0.15, 0.15, 0.15])
            landmark.size = 0.03
            # world.landmarks[i].state.p_pos = position[i]
            # world.landmarks[i].state.p_vel = np.zeros(world.dim_p)

        return world

    def reset_world(self, world, agent_id, step = 0):
        for i in range(self.n_landmarks):
            world.landmarks[i].state.p_pos = np.random.uniform(-self.ratio*1, self.ratio*1, world.dim_p)
            world.landmarks[i].state.p_vel = np.zeros(world.dim_p)
        all_pos = np.random.uniform(-self.ratio*1, +self.ratio*1, (self.num_agents, world.dim_p))
        for i in range(self.num_agents):
            world.agents[i].state.p_pos = all_pos[i]
            world.agents[i].state.p_vel = np.random.uniform(-1-self.ratio, 1+self.ratio, world.dim_p)
        # Calculate the distance between the agent and all agents
        distance = np.linalg.norm(all_pos - all_pos[agent_id],axis=1)
        action_agents = list(np.where(distance<=2*(self.prosp_dist+self.good_neigh_dist))[0])
        neighbors = list(np.where(distance<=self.good_neigh_dist)[0])
        return action_agents, neighbors


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, index, world):
        agent = world.agents[index]
        rew = 0
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        rew -= min(dists)

        # for food in world.landmarks:
        #     if self.is_collision(food, agent):
        #         rew += 1

        for a in world.agents:
            if a == agent: continue
            if self.is_collision(a, agent):
                rew -= 1
        return rew


    def observation(self, index, world):
        # current agent
        agent = world.agents[index]
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
