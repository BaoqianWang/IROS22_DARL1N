import numpy as np
from mpe_local.multiagent.core_neighbor import World, Agent, Landmark
from mpe_local.multiagent.scenario import BaseScenario
import os


class Scenario(BaseScenario):
    def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, good_sight, adv_sight, no_wheel, ratio, prosp, max_good_neighbor, max_adv_neighbor):
        self.n_good = n_good
        self.n_adv = n_adv
        self.n_landmarks = n_landmarks
        self.n_food = n_food
        self.n_forests = n_forests
        self.num_agents = n_adv + n_good
        self.alpha = alpha
        #self.good_sight = good_sight
        #self.adv_sight = adv_sight
        self.no_wheel = no_wheel
        self.size_food = ratio
        self.size = ratio
        self.ratio = ratio
        self.good_neigh_dist = good_sight
        self.adv_neigh_dist = adv_sight
        self.max_good_neighbor = max_good_neighbor
        self.max_adv_neighbor = max_adv_neighbor
        self.prosp_dist = prosp
        # print(sight,"sight___wolf_sheep_v2")
        # print(alpha,"alpha######################")

    def make_world(self):
        world = World()
        # set any world properties first
        world.collaborative = False
        world.dim_c = 2
        world.size = self.ratio
        #world.sight = self.sight
        num_good_agents = self.n_good
        num_adversaries = self.n_adv
        world.num_good_agents = num_good_agents
        world.num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        self.num_agents = num_agents
        num_landmarks = self.n_landmarks
        num_food = self.n_food
        num_forests = self.n_forests

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
            agent.adversary = True if i < num_adversaries else False
            agent.size = (0.075 if agent.adversary else 0.05)
            agent.accel = (2.0 if agent.adversary else 4.0)
            if agent.adversary:
                agent.showmore = np.zeros(num_good_agents)
            else:
                agent.showmore = np.zeros(num_food)
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = (2 if agent.adversary else 3)
            agent.live = 1

        # make initial conditions
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.boundary = False

        world.landmarks = world.food
        for i in range(num_agents):
            self.reset_world(world, i)
        return world


    def reset_world(self, world, agent_id, step = 0):
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.65, 0.15])
            landmark.state.p_pos = np.random.uniform(-self.ratio*1, self.ratio*1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        all_pos = np.random.uniform(-self.ratio*1, +self.ratio*1, (self.num_agents, world.dim_p))

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.45, 0.95]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.live = 1
            agent.state.p_pos = all_pos[i]
            agent.state.p_vel = np.random.uniform(-1-self.ratio, 1+self.ratio, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if agent.adversary:
                agent.showmore = np.zeros(world.num_good_agents)
            else:
                agent.showmore = np.zeros(world.num_adversaries)

        distance = np.linalg.norm(all_pos - all_pos[agent_id],axis=1)
        action_agents = list(np.where(distance<=2*(self.prosp_dist+self.good_neigh_dist))[0])
        neighbors = list(np.where(distance<=self.good_neigh_dist)[0])
        neighbors.remove(agent_id)
        return action_agents, neighbors


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
        if agent in self.adversaries(world):
            for ag in self.good_agents(world):
                if ag.live:
                    return 0
            return 1
        else:
            if not agent.live:
                return 1
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

    def reward(self, index, world):
        agent = world.agents[index]
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        # Live agent:
        rew = 0
        if agent.live:
            # Good agent:
            if not agent.adversary:
                for other_id in agent.neighbors:
                    # Eaten by wolf
                    if self.is_collision(agent, world.agents[other_id]) and world.agents[other_id].adversary:
                        rew -= 5
                        agent.live = False
                distance_min = min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])
                rew -= distance_min
                for food in world.food:
                    if self.is_collision(agent, food):
                        food.state.p_pos = np.random.uniform(-1*self.ratio, 1*self.ratio, world.dim_p)
                        rew += 20

            # Adv agent:
            else:
                for other_id in agent.neighbors:
                    if not world.agents[other_id].live: continue
                    # Eat sheep
                    if self.is_collision(agent, world.agents[other_id]) and not world.agents[other_id].adversary:
                        rew += 15

                # Distance to cloest sheep
                dist2good = [np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[other_id].state.p_pos))) for other_id in agent.neighbors if not world.agents[other_id].adversary]
                if len(dist2good) > 0:
                    rew -= min(dist2good)

        return rew


    def observation(self, index, world):
        agent = world.agents[index]

        if agent.adversary:
            max_neighbor = self.max_adv_neighbor
            neighbor_sight = self.adv_neigh_dist
        else:
            max_neighbor = self.max_good_neighbor
            neighbor_sight = self.good_neigh_dist

        # get positions of all entities in this agent's reference frame
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

        for i, other in enumerate(world.agents):
            if other is agent: continue
            distance = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if distance <= neighbor_sight and num_neighbor < max_neighbor-1:
                other_vel[num_neighbor] = other.state.p_vel
                other_pos[num_neighbor] = other.state.p_pos- agent.state.p_pos
                other_live[num_neighbor] = [other.live]
                num_neighbor += 1
        # print(result.shape,"shape#################")
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [np.array([agent.live])] + entity_pos + other_pos + other_vel + other_live)
