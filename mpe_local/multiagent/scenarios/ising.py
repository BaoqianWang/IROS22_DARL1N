import numpy as np
from mpe_local.multiagent.core_ising import IsingWorld, IsingAgent


class Scenario():
  def __init__(self, n_good, n_adv, n_landmarks, n_food, n_forests, alpha, good_sight, adv_sight, no_wheel, ratio,  max_good_neighbor, max_adv_neighbor):
    self.n_good = n_good
    self.n_landmarks = n_landmarks
    self.n_food = n_food
    self.n_adv = n_adv
    self.n_forests = n_forests
    self.alpha = alpha
    self.sight = good_sight
    self.no_wheel = no_wheel
    self.good_neigh_dist = good_sight
    self.adv_neigh_dist = adv_sight
    self.num_agents = self.n_good + self.n_adv
    self.max_good_neighbor = max_good_neighbor
    self.max_adv_neighbor = max_adv_neighbor
    #print(self.num_agents)


  def _calc_mask(self, agent, shape_size):
    # compute the neighbour mask for each agent
    if agent.view_sight == -1:
      # fully observed
      agent.spin_mask += 1
    elif agent.view_sight == 0:
      # observe itself
      agent.spin_mask[agent.state.id] = 1
    elif agent.view_sight > 0:
      # observe neighbours
      delta = list(range(-int(agent.view_sight), int(agent.view_sight) + 1, 1))
      delta.remove(0)  # agent itself is not counted as neighbour of itself
      for dt in delta:
        row = agent.state.p_pos[0]
        col = agent.state.p_pos[1]
        row_dt = row + dt
        col_dt = col + dt
        if row_dt in range(0, shape_size):
          agent.spin_mask[agent.state.id + shape_size * dt] = 1
        if col_dt in range(0, shape_size):
          agent.spin_mask[agent.state.id + dt] = 1

      # the graph is cyclic, most left and most right are neighbours
      if agent.state.p_pos[0] < agent.view_sight:
        tar = shape_size - (np.array(
          range(0, int(agent.view_sight - agent.state.p_pos[0]), 1)) + 1)
        tar = tar * shape_size + agent.state.p_pos[1]
        agent.spin_mask[tar] = [1] * len(tar)

      if agent.state.p_pos[1] < agent.view_sight:
        tar = shape_size - (np.array(
          range(0, int(agent.view_sight - agent.state.p_pos[1]), 1)) + 1)
        tar = agent.state.p_pos[0] * shape_size + tar
        agent.spin_mask[tar] = [1] * len(tar)

      if agent.state.p_pos[0] >= shape_size - agent.view_sight:
        tar = np.array(
          range(0, int(agent.view_sight - (shape_size - 1 -
                                           agent.state.p_pos[0])),
                1)
        )
        tar = tar * shape_size + agent.state.p_pos[1]
        agent.spin_mask[tar] = [1] * len(tar)

      if agent.state.p_pos[1] >= shape_size - agent.view_sight:
        tar = np.array(
          range(0, int(agent.view_sight - (shape_size - 1 -
                                           agent.state.p_pos[1])),
                1)
        )
        tar = agent.state.p_pos[0] * shape_size + tar
        agent.spin_mask[tar] = [1] * len(tar)

  def make_world(self):
    world = IsingWorld()
    world.agent_view_sight = 1
    world.dim_spin = 2
    world.dim_pos = 2
    num_agents = self.num_agents
    world.n_agents = num_agents
    world.shape_size = int(np.ceil(np.power(num_agents, 1.0 / world.dim_pos)))
    world.global_state = np.zeros((world.shape_size,) * world.dim_pos)
    # assume 0 external magnetic field
    world.field = np.zeros((world.shape_size,) * world.dim_pos)

    world.agents = [IsingAgent(view_sight=world.agent_view_sight)
                    for i in range(num_agents)]
    world.ising = True
    world.max_good_neighbor = self.max_good_neighbor
    # make initial conditions
    self.reset_world(world)

    return world

  def reset_world(self, world):

    world_mat = np.array(
      range(np.power(world.shape_size, world.dim_pos))). \
      reshape((world.shape_size,) * world.dim_pos)
    # init agent state and global state
    for i, agent in enumerate(world.agents):
      agent.name = 'agent %d' % i
      agent.color = np.array([0.35, 0.35, 0.85])
      agent.state.id = i
      agent.state.p_pos = np.where(world_mat == i)
      agent.state.spin = np.random.choice(world.dim_spin)
      agent.spin_mask = np.zeros(world.n_agents)

      assert world.dim_pos == 2, "cyclic neighbour only support 2D now"
      self._calc_mask(agent, world.shape_size)
      world.global_state[agent.state.p_pos] = agent.state.spin

    n_ups = np.count_nonzero(world.global_state.flatten())
    n_downs = world.n_agents - n_ups
    world.order_param = abs(n_ups - n_downs) / (world.n_agents + 0.0)

  def reward(self, agent, world):
    # turn the state into -1/1 for easy computing
    world.global_state[np.where(world.global_state == 0)] = -1

    mask_display = agent.spin_mask.reshape((int(np.sqrt(world.n_agents)), -1))

    local_reward = - 0.5 * world.global_state[agent.state.p_pos] \
                   * np.sum(world.global_state.flatten() * agent.spin_mask)
    # print('index', agent.state.id)
    # print('state', world.global_state)
    # print('mask', agent.spin_mask)
    # print('reward', -local_reward[0])
    world.global_state[np.where(world.global_state == -1)] = 0

    #print(world.global_state[agent.state.p_pos])
    return -local_reward[0]

  def observation(self, agent, world):
    ret = [world.global_state[agent.state.p_pos][0]]
    ret +=  world.global_state.flatten()[np.where(agent.spin_mask == 1)].tolist()
    #ret.append(world.global_state[agent.state.p_pos][0])
    ret = np.asarray(ret)
    return ret

  def info(self, agent, world):
    return 0

  def done(self, agent, world):
    return False
