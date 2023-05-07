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
            np.ones(batch_size),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), batch_inds)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
    
    
class PriorityBuffer(BaseBuffer):
    def __init__(self, params: PriorityBufferOptions, *args, max_steps, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.params = params
        
        self.n_steps_total = max_steps
        # Kept just at the max of the seen errors
        self.max_prio_td = 1
        
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=self.obs_dtype)
    
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=self.act_dtype)

        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.td_errors = np.zeros((self.buffer_size), dtype=np.float32)
        
        # We need this because we're going to be doing a bunch of sorting
        # This array will store the age of each sample at an index
        self.sample_locs = np.array(range(self.buffer_size), dtype=int)
        
        # How we define priorities depends on the type
        if self.params.prio_type == PriorityType.rank:
            self.prios = (1 / np.arange(1, self.buffer_size+1, 1)) ** self.params.alpha
        elif self.params.prio_type == PriorityType.proportional:
            self.prios = np.zeros((self.buffer_size), dtype=np.float32)
        else:
            raise ValueError(f"Invalid priority replay buffer type: {self.params.prio_type}")
        
        self.buffer_len = 0
        
    @property
    def sample_prob(self) -> np.ndarray:
        """Probability distribution to use for sampling"""
        if self.params.prio_type == PriorityType.proportional:
            # Only update when we have to
            self.prios[:self.buffer_len] = self.td_errors[:self.buffer_len] ** self.params.alpha
        return self.prios[:self.buffer_len] / np.sum(self.prios[:self.buffer_len])
    
    def size(self) -> int:
        return self.buffer_len
        
    def reset(self) -> None:
        # To reset the buffer, set the length back to 0
        self.buffer_len = 0
        
    def sort_transitions(self):
        # Arranged order of elements w.r.t. td errors
        # Keeps things easy to work with when it comes to prios
        idx = np.argsort(self.td_errors[:self.buffer_len])
        
        self.observations[:self.buffer_len] = self.observations[:self.buffer_len][idx]
        self.next_observations[:self.buffer_len] = self.next_observations[:self.buffer_len][idx]
        self.actions[:self.buffer_len] = self.actions[:self.buffer_len][idx]
        self.rewards[:self.buffer_len] = self.rewards[:self.buffer_len][idx]
        self.dones[:self.buffer_len] = self.dones[:self.buffer_len][idx]
        self.td_errors[:self.buffer_len] = self.td_errors[:self.buffer_len][idx]
        
        self.sample_locs[:self.buffer_len][idx]

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info
    ) -> None:
        # Index of oldest transition
        if self.buffer_len < self.buffer_size:
            # If not full, just keep filling
            loc = self.buffer_len
            # loc = self.sample_locs[self.buffer_len]
        else:
            # Decrement as the max should always be buffer_size-1
            self.sample_locs -= 1
            # Oldest element is lowest index
            loc = np.argmin(self.sample_locs)
            # Could also do (self.sample_locs < 0)[0] 
            # (might be faster), as one 1 element ever less than 0 here
        
        # Copy to avoid modification by reference
        self.observations[loc] = np.array(obs).copy()
        self.next_observations[loc] = np.array(next_obs).copy()
        
        self.actions[loc] = np.array(action).copy()
        self.rewards[loc] = np.array(reward).copy()
        self.dones[loc] = np.array(done).copy()
        
        self.td_errors[loc] = self.max_prio_td
        
        if self.buffer_len < self.buffer_size:
            # sample_locs are preset to be a range, so no update needed
            self.buffer_len += 1
        else:
            # New transition is the freshest
            self.sample_locs[loc] = self.buffer_size-1
        
        # Re-sort transitions
        self.sort_transitions()
    
    def compute_IS_weights(self, sample_prs, batch_inds, step) -> np.ndarray:
        """Compute importance sampling weight used to de-bias the gradient estimate"""
        # Beta increases to 1 over the training
        beta = self.params.beta + (1-self.params.beta) * step / self.n_steps_total
        weights = (1/(self.buffer_size*sample_prs[batch_inds]))**beta
        weights_normalised = weights / np.max(weights)
        return weights_normalised
    
    def sample(self, batch_size: int, step: int) -> ReplayBufferSamples:
        sample_prs = self.sample_prob
        batch_inds = np.random.choice(self.buffer_len, size=batch_size, p=sample_prs)
        
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
            self.compute_IS_weights(sample_prs, batch_inds, step),
        )
        
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), batch_inds)
    
    def update_td_errors(self, td_errors, batch_inds):
        new_td_errors = np.array(np.abs(td_errors)).copy()
        self.td_errors[batch_inds] = new_td_errors
        self.max_prio_td = np.max(np.concatenate((new_td_errors, np.array([self.max_prio_td]))))
        self.sort_transitions()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)
        
        


class BufferType(EnumByName, CallableEnum):
    random = partial(RandomBuffer)
    priority = partial(PriorityBuffer)
