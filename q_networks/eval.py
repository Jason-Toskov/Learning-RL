from random import randint

import cv2
import numpy as np
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from q_networks.configs.params import DQNParams
from q_networks.utils.helpers import obs_to_tensor
from q_networks.utils.models import QNetwork


class Evaluator:
    def __init__(self, cfg: DQNParams, tb: SummaryWriter):
        self.cfg = cfg
        self.tb = tb
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        self.vec_env = make_atari_env(f"{self.cfg.env.type.value}")
        self.env = VecFrameStack(self.vec_env, n_stack=self.cfg.env.history)
        self.step = None
        self.last_obs = None
        
        
    def run_evaluation(self, policy_net: QNetwork, global_step: int):
        self.eval_episodes(policy_net, global_step)
        self.record_episode(policy_net, global_step)
        
    def eval_episodes(self, policy_net: QNetwork, global_step: int):
        policy_net.eval()
        reward_list = []
        value_estimates = []
        for ep_num in tqdm(range(self.cfg.eval.n_episodes), desc="Eval episode", leave=False):
            self.last_obs = self.env.reset()
            noop_steps = randint(0,self.cfg.eval.noop_max)
            self.step = 0
            reward_accumulator = 0
            while self.step < self.cfg.eval.ep_max_frames:
                if self.step < noop_steps:
                    # NOOP action is '0'
                    action = np.array([0])
                else:
                    # We act epsilon greedily with a fixed eps
                    if np.random.rand() < self.cfg.eval.eps:
                        action = np.array([self.env.action_space.sample()])
                    else:
                        obs = obs_to_tensor(self.last_obs).to(self.device)
                        with torch.no_grad():
                            action_values = policy_net(obs)
                            value, action = torch.max(action_values, dim=1)
                            value_estimates.append(value.cpu().item()) # Value of current state
                            action = action.reshape(-1).cpu().numpy() # Action to take
                
                new_obs, rewards, dones, infos = self.env.step(action)
                self.last_obs = new_obs
                reward_accumulator += rewards.item()
                self.step += 1
                
                if dones[0]:
                    reward_list.append(reward_accumulator)
                    break
        
        mean_ep_reward = np.mean(reward_list)
        self.tb.add_scalar("eval/average_reward", mean_ep_reward, global_step)
        
        avg_value_est = np.mean(value_estimates)
        self.tb.add_scalar("eval/value_estimate", avg_value_est, global_step)
    
    def obs_to_frame(self, obs):
        obs = np.expand_dims(obs[0], -1)
        frame = torch.tensor(cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB))
        return frame
    
    def record_episode(self, policy_net: QNetwork, global_step: int):
        self.last_obs = self.env.reset()
        for i in range(self.cfg.env.history):
            frames = [self.obs_to_frame(self.last_obs[..., i])]
        while self.step < self.cfg.eval.ep_max_frames:
            if np.random.rand() < self.cfg.eval.eps:
                action = np.array([self.env.action_space.sample()])
            else:
                obs = obs_to_tensor(self.last_obs).to(self.device)
                with torch.no_grad():
                    action = policy_net(obs).argmax(dim=1).reshape(-1).cpu().numpy()
                    
            new_obs, rewards, dones, infos = self.env.step(action)
            self.last_obs = new_obs
            frames.append(self.obs_to_frame(self.last_obs[..., -1]))
            
            if dones[0]:
                video_frames = torch.stack(frames).unsqueeze(0).permute(0,1,4,2,3)
                self.tb.add_video("eval/trajectory", video_frames, global_step, fps=15)
                break
        
    def run_random_agent(self):
        pass