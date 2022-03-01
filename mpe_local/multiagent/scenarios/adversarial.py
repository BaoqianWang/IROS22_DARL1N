import numpy as np
from mpe_local.multiagent.core import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os
import random
SIGHT = 0.5


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
        world.size = self.ratio
        world.good_neigh_dist = self.good_neigh_dist
        world.adv_neigh_dist = self.adv_neigh_dist
        world.max_good_neighbor = self.max_good_neighbor
        world.max_adv_neighbor = self.max_adv_neighbor
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.n_good
        num_adversaries = self.n_adv
        world.num_good_agents = num_good_agents
        world.num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = self.n_landmarks
        num_food = self.n_food
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.08 if agent.adversary else 0.08
            agent.accel = 4.0 if agent.adversary else 4.0
            if agent.adversary:
                agent.showmore = np.zeros(num_good_agents)
            else:
                agent.showmore = np.zeros(num_food)
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 3 if agent.adversary else 3
            agent.live = 1


        # make initial conditions
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.sight = 1
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False

        world.landmarks =  world.food

        self.reset_world(world)
        return world


    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.45, 0.95]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.live = 1
            if agent.adversary:
                agent.showmore = np.zeros(world.num_good_agents)
            else:
                agent.showmore = np.zeros(world.num_adversaries)
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.65, 0.15])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1*self.ratio, +1*self.ratio, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1*self.ratio, 1*self.ratio, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def done(self, agent, world):
        return 0

    def info(self, agent, world):
        time_grass = []
        time_live = []

        mark_grass = 0
        if agent.live:
            time_live.append(1)
            for food in world.food:
                if self.is_collision(agent, food):
                    mark_grass = 1
                    break
        else:
            time_live.append(0)
        if mark_grass:
            time_grass.append(1)
        else:
            time_grass.append(0)

        return np.concatenate([np.array(time_grass)]+[np.array(time_live)])


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        rew = 0

        if agent.live:
            # Good agent
            dist2food = min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])
            rew -= dist2food
            for food in world.food:
                if self.is_collision(agent, food):
                    rew += 10
                    food.state.p_pos = np.random.uniform(-1*self.ratio, 1*self.ratio, world.dim_p)

            if not agent.adversary: # Good agents
                num_collide = 0
                good_collide = 0
                for other_agent in world.agents:
                    if other_agent == agent:continue
                    if not other_agent.live:continue
                    if self.is_collision(agent, other_agent) and other_agent.adversary: # Collide with one agent of another side
                        num_collide += 1
                        good_collide += 1
                        for other_good_agent in world.agents:
                            if not other_good_agent.live: continue
                            if other_good_agent == other_agent: continue
                            if other_good_agent == agent: continue
                            if other_good_agent.adversary: continue
                            if self.is_collision(other_good_agent, other_agent):
                                good_collide += 1
                                rew += 5
                                good_collide = 0

                if num_collide >= 2:
                    agent.live = False
                    rew -= 5

                distance_min = [np.sqrt(np.sum(np.square(agent.state.p_pos - other_agent.state.p_pos))) for other_agent in world.agents if other_agent.adversary and other_agent.live]
                if (len(distance_min) > 0):
                    rew -= min(distance_min)

            if agent.adversary:
                num_collide = 0
                good_collide = 0
                for other_agent in world.agents:
                    if other_agent == agent:continue
                    if not other_agent.live:continue
                    if self.is_collision(agent, other_agent) and not other_agent.adversary:
                        num_collide += 1
                        good_collide += 1
                        for other_good_agent in world.agents:
                            if not other_good_agent.live: continue
                            if other_good_agent== other_agent: continue
                            if other_good_agent== agent: continue
                            if not other_good_agent.adversary: continue
                            if self.is_collision(other_good_agent, other_agent):
                                good_collide += 1
                                rew += 5
                                good_collide = 0

                if num_collide >= 2:
                    agent.live = False
                    rew -= 5

                distance_min = [np.sqrt(np.sum(np.square(agent.state.p_pos - other_agent.state.p_pos))) for other_agent in world.agents if not other_agent.adversary and other_agent.live]
                if (len(distance_min) > 0):
                    rew -= min(distance_min)

        return rew



    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        if agent.adversary:
            max_neighbor = self.max_adv_neighbor
            neighbor_sight = self.adv_neigh_dist
        else:
            max_neighbor = self.max_good_neighbor
            neighbor_sight = self.good_neigh_dist

        dist = []
        for i, landmark in enumerate(world.landmarks):
            dist.append((i, np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
        dist = sorted(dist, key = lambda t: t[1])
        entity_pos = []
        for i, land_dist in dist:
            entity_pos.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
            entity_pos.append([0])

        for j in range(int(max_neighbor/2) - len(world.landmarks)):
            entity_pos.append([0, 0])
            entity_pos.append([0])

        other_pos = [[0, 0] for i in range(max_neighbor-1)]
        other_live = [[0] for i in range(max_neighbor-1)]
        other_vel = [[0, 0] for i in range(max_neighbor-1)]
        num_neighbor = 0
        no_live = False
        for i, other in enumerate(world.agents):
            if other is agent: continue
            distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if distance <= neighbor_sight and num_neighbor < max_neighbor-1:
                other_vel[num_neighbor] = other.state.p_vel
                other_pos[num_neighbor] = other.state.p_pos- agent.state.p_pos
                other_live[num_neighbor] = [other.live]
                no_live = no_live or other.live
                num_neighbor += 1

        #agent.live = 1
        #other_live[-1] = [1]



        #print('observation', result)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live)
