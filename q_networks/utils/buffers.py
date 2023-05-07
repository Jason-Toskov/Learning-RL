from abc import ABC,abstractmethod
import random
from typing import NamedTuple, Union
from functools import partial

import numpy as np
import torch
from numpy._typing import DTypeLike, _ShapeLike
from pydantic import BaseModel

from q_networks.configs.enums import EnumByName, CallableEnum
from q_networks.utils.helpers import SumTree


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: torch.Tensor
    batch_idx: np.ndarray

# TODO: move these to their own file like `buffer_params.py`
class PriorityType(EnumByName):
    proportional = 0
    rank = 1


class BaseBufferOptions(BaseModel):
    size: int


class RandomBufferOptions(BaseBufferOptions):
    pass


class PriorityBufferOptions(BaseBufferOptions):
    eps: float
    alpha: float 
    beta: float
    prio_type: PriorityType


class BaseBuffer(ABC):
    def __init__(
        self, 
        params: BaseBufferOptions, 
        obs_shape: _ShapeLike,
        obs_dtype: DTypeLike,
        action_dim: _ShapeLike,
        act_dtype: DTypeLike,
        device: Union[torch.device, str] = "auto",
        **kwargs
    ):
        self.device = device
        
        self.params = params
        self.buffer_size = self.params.size
        
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.act_dtype = act_dtype
    
    @abstractmethod
    def size(self) -> int:
        """Should return the current buffer size"""
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Should reset and clear the buffer"""
        pass
    
    @abstractmethod
    def add(self, obs, next_obs, action, reward, done, info) -> None:
        """Add a transition to the buffer"""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int, step: int)  -> ReplayBufferSamples:
        """Should return a batch of samples"""
        pass


class RandomBuffer(BaseBuffer):
    def __init__(self, params: RandomBufferOptions, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
    
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=self.act_dtype)

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        
        # These are used to make the buffer circular
        # When pos reaches buffer_size, it resets to 0
        self.pos = 0
        self.full = False
        
    def size(self) -> int:
        if self.full:
            return self.buffer_size
        return self.pos
    
    def reset(self) -> None:
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def sample(self, batch_size: int, step: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
            np.ones(batch_size).reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), batch_inds)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    

# Taken from: https://github.com/Howuhh/prioritized_experience_replay
class PriorityBuffer(BaseBuffer):
    def __init__(self, params: PriorityBufferOptions, *args, max_steps, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params = params
        
        self.tree = SumTree(size=self.buffer_size)
        
        self.n_steps_total = max_steps
        self.max_prio = self.params.eps
        
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=self.act_dtype)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        
        self.count = 0
        self.real_size = 0
           
    def reset(self):
        self.tree = SumTree(size=self.buffer_size)
        self.max_prio = self.params.eps
        self.count = 0
        self.real_size = 0
        
    def size(self):
        return self.real_size
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info
    ) -> None:
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_prio, self.count)
        
        # Copy to avoid modification by reference
        self.observations[self.count] = np.array(obs).copy()
        self.next_observations[self.count] = np.array(next_obs).copy()
        self.actions[self.count] = np.array(action).copy()
        self.rewards[self.count] = np.array(reward).copy()
        self.dones[self.count] = np.array(done).copy()
        
        # update counters
        self.count = (self.count + 1) % self.buffer_size
        self.real_size = min(self.buffer_size, self.real_size + 1)
        
    def sample(self, batch_size: int, step: int):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"
    
        sample_idxs, tree_idxs = [], []
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        
        probs = priorities / self.tree.total
        
        beta = self.params.beta + (1-self.params.beta) * step / self.n_steps_total
        weights = (self.real_size * probs) ** -beta
        weights = weights / weights.max()
        
        data = (
            self.observations[sample_idxs, :],
            self.actions[sample_idxs, :],
            self.next_observations[sample_idxs, :],
            self.dones[sample_idxs].reshape(-1, 1),
            self.rewards[sample_idxs].reshape(-1, 1),
            weights.reshape(-1, 1)
        )        
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), tree_idxs)
    
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.params.eps) ** self.params.alpha

            self.tree.update(data_idx, priority)
            self.max_prio = max(self.max_prio, priority)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
        

class BufferType(EnumByName, CallableEnum):
    random = partial(RandomBuffer)
    priority = partial(PriorityBuffer)
