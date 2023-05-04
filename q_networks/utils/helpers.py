import numpy as np
import torch


def obs_to_tensor(obs):
    if isinstance(obs, np.ndarray):
        obs = torch.tensor(obs)
    obs = torch.permute(obs, (0,3,1,2))
    obs = (obs.float() / 255)
    return obs