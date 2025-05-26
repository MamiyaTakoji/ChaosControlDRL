from xuance.common.memory_tools import Buffer
import random
import numpy as np
from gym import Space
from abc import ABC, abstractmethod
from xuance.common import Optional, Union
from xuance.common import discount_cumsum
from xuance.common.segtree_tool import SumSegmentTree, MinSegmentTree
from collections import deque
from xuance.common import Dict
def space2shape(observation_space):
    """Convert gym.space variable to shape
    Args:
        observation_space: the space variable with type of gym.Space.

    Returns:
        The shape of the observation_space.
    """
    if isinstance(observation_space, Dict) or isinstance(observation_space, dict):
        return {key: observation_space[key].shape for key in observation_space.keys()}
    elif isinstance(observation_space, tuple):
        return observation_space
    else:
        return observation_space.shape
def create_memory(shape: Optional[Union[tuple, dict]],
                  n_envs: int,
                  n_size: int,
                  dtype: type = np.float32):
    """
    Create a numpy array for memory data.

    Args:
        shape: data shape.
        n_envs: number of parallel environments.
        n_size: length of data sequence for each environment.
        dtype: numpy data type.

    Returns:
        An empty memory space to store data. (initial: numpy.zeros())
    """
    if shape is None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in shape.items():
            if value is None:  # save an object type
                memory[key] = np.zeros([n_envs, n_size], dtype=object)
            else:
                memory[key] = np.zeros([n_envs, n_size] + list(value), dtype=dtype)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([n_envs, n_size] + list(shape), dtype)
    else:
        raise NotImplementedError
def store_element(data: Optional[Union[np.ndarray, dict, float]],
                  memory: Union[dict, np.ndarray],
                  ptr: int):
    """
    Insert a step of data into current memory.

    Args:
        data: target data that to be stored.
        memory: the memory where data will be stored.
        ptr: pointer to the location for the data.
    """
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in data.items():
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data
def sample_batch(memory: Optional[Union[np.ndarray, dict]],
                 index: Optional[Union[np.ndarray, tuple]]):
    """
    Sample a batch of data from the selected memory.

    Args:
        memory: memory that contains experience data.
        index: pointer to the location for the selected data.

    Returns:
        A batch of data.
    """
    if memory is None:
        return None
    elif isinstance(memory, dict):
        batch = {}
        for key, value in memory.items():
            batch[key] = value[index]
        return batch
    else:
        return memory[index]
class DummyOffPolicyBuffer(Buffer):
    """
    Replay buffer for off-policy DRL algorithms.

    Args:
        observation_space: the observation space of the environment.
        action_space: the action space of the environment.
        auxiliary_shape: data shape of auxiliary information (if exists).
        n_envs: number of parallel environments.
        buffer_size: the total size of the replay buffer.
        batch_size: size of transition data for a batch of sample.
    """

    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 n_envs: int,
                 buffer_size: int,
                 batch_size: int):
        super(DummyOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
        self.n_envs, self.batch_size = n_envs, batch_size
        assert buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = buffer_size // self.n_envs
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)

    def clear(self):
        self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
        self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
        self.rewards = create_memory((), self.n_envs, self.n_size)
        self.terminals = create_memory((), self.n_envs, self.n_size)

    def store(self, obs, acts, rews, terminals, next_obs):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(next_obs, self.next_observations, self.ptr)
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def sample(self, batch_size=None):
        bs = self.batch_size if batch_size is None else batch_size
        env_choices = np.random.choice(self.n_envs, bs)
        step_choices = np.random.choice(self.size, bs)

        samples_dict = {
            'obs': sample_batch(self.observations, tuple([env_choices, step_choices])),
            'actions': sample_batch(self.actions, tuple([env_choices, step_choices])),
            'obs_next': sample_batch(self.next_observations, tuple([env_choices, step_choices])),
            'rewards': sample_batch(self.rewards, tuple([env_choices, step_choices])),
            'terminals': sample_batch(self.terminals, tuple([env_choices, step_choices])),
            'batch_size': bs,
        }
        return samples_dict

def Test():
    from gym.spaces import MultiDiscrete, Space, Discrete, Box
    action_space = MultiDiscrete([2,3,4])
    M = create_memory(shape = space2shape(action_space),
                      n_envs = 5,
                      n_size = 1000)
    return M
if __name__ == "__main__":
    M = Test()


























