from abc import ABC,abstractmethod 
from typing import NamedTuple, Union
from functools import partial

import numpy as np
import torch
from numpy._typing import DTypeLike, _ShapeLike
from pydantic import BaseModel

from q_networks.configs.enums import EnumByName, CallableEnum

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

# TODO: move these to their own file like `buffer_params.py`
class BaseBufferOptions(BaseModel):
    size: int
    
class RandomBufferOptions(BaseBufferOptions):
    pass

class PriorityBufferOptions(BaseBufferOptions):
    alpha: float 
    beta: float


class BaseBuffer(ABC):
    def __init__(
        self, 
        params: BaseBufferOptions, 
        obs_shape: _ShapeLike,
        obs_dtype: DTypeLike,
        action_dim: _ShapeLike,
        act_dtype: DTypeLike,
        device: Union[torch.device, str] = "auto"
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
    def sample(self, batch_size):
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
    
    def sample(self, batch_size: int):
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
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    
    
class PriorityBuffer(BaseBuffer):
    pass


class BufferType(EnumByName, CallableEnum):
    random = partial(RandomBuffer)
    priority = partial(PriorityBuffer)
