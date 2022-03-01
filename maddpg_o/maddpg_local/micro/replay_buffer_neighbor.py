import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, num_agents, max_neighbors, agent_index):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.num_agents = num_agents
        self.max_neighbors = max_neighbors
        self.agent_index = agent_index

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs, action_n, new_obs, target_action_n, rew):
        data = (obs, action_n, new_obs, target_action_n, rew)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, agents):
        #obss, action_ns, new_obss, target_action_ns, rews = [], [], [], [], []
        target_action_ns = [[] for i in range(self.max_neighbors)]
        action_ns = [[] for i in range(self.max_neighbors)]
        obss = [[]]
        new_obss = [[]]
        rews = []

        for i in idxes:
            data = self._storage[i]
            obs, action_n, new_obs, target_action_n, rew = data
            obss[0].append(obs.tolist())
            new_obss[0].append(new_obs.tolist())

            for j in range(self.max_neighbors):
                target_action_ns[j].append(target_action_n[j].tolist())
                action_ns[j].append(action_n[j].tolist())

            rews.append(rew)

        target_action_array = [np.array(value) for value in target_action_ns]
        #print(target_action_array)
        action_array = [np.array(value) for value in action_ns]
        #print(action_array)
        obss_array = [np.array(value) for value in obss]
        new_obss_array = [np.array(value) for value in new_obss]

        return obss_array, action_array, new_obss_array, target_action_array,  np.array(rews)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes, agents):
        return self._encode_sample(idxes, agents)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
