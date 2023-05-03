import yaml

import numpy as np
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.nn import functional as F
from tqdm import tqdm

from q_networks.utils.buffers import RandomBuffer
from q_networks.utils.models import QNetwork
from q_networks.configs.params import DQNParams


class DQNTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = DQNParams.parse_obj(yaml.safe_load(f))
        
        self.vec_env = make_atari_env(f"{self.cfg.env.type.value}")
        self.env = VecFrameStack(self.vec_env, n_stack=self.cfg.env.history)
        self.obs_shape = self.env.observation_space.shape
        self.obs_dtype = self.env.observation_space.dtype
        self.action_dim = 1
        self.action_len = self.env.action_space.n
        self.action_dtype = self.env.action_space.dtype
        self.last_obs = self.env.reset()
        
        self.step = 0
        self.episode_num = 0
        self.pbar = tqdm(range(self.cfg.train.max_steps))
        
        self.eps = self.cfg.eps.start
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.policy_net = QNetwork(self.cfg.env.history, self.action_len).to(self.device)
        self.target_net = QNetwork(self.cfg.env.history, self.action_len).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = RandomBuffer(
            buffer_size = self.cfg.buffer.size,
            obs_shape = self.obs_shape,
            obs_dtype = self.obs_dtype,
            action_dim = self.action_dim,
            act_dtype = self.action_dtype,
            device = self.device
        )
        
        self.episode_reward = [0]
        self.losses = []
        
        self.optim = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.cfg.train.lr,
        )
        
    def obs_to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        obs = torch.permute(obs, (0,3,1,2))
        obs = (obs.float() / 255)
        return obs
        
    def sample_actions(self, random=False):
        if self.step < self.cfg.train.start_step or random or np.random.rand() < self.eps:
            action = np.array([self.env.action_space.sample()])
        else:
            obs = self.obs_to_tensor(self.last_obs).to(self.device)
            with torch.no_grad():
                action = self.policy_net(obs).argmax(dim=1).reshape(-1).cpu().numpy()
        
        return action
    
    def end_step(self):
        if self.step % self.cfg.train.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.eps = self.cfg.eps.schedule(
            self.step, 
            self.cfg.eps.start, 
            self.cfg.eps.end, 
            self.cfg.eps.steps
        )
        
        self.step += 1
        
        self.pbar.update(1)
        
    def run_rollouts(self):
        self.policy_net.eval()
        for rollout_step in range(self.cfg.train.update_freq):
            
            action = self.sample_actions()
            
            new_obs, rewards, dones, infos = self.env.step(action)
            
            self.buffer.add(
                self.last_obs,
                new_obs,
                action,
                rewards,
                dones,
                infos
            )
            
            self.last_obs = new_obs
            
            self.end_step()
            
            self.episode_reward[-1] += rewards[0]
            for idx, done in enumerate(dones):
                if done:
                    self.episode_num += 1
                    self.last_obs = self.env.reset()
                    
                    if len(self.episode_reward) < 10:
                        self.episode_reward.append(0)
                    else:
                        print(np.mean(self.episode_reward))
                        with open("./rewards.txt","a") as f:
                            if len(self.losses) == 0:
                                self.losses = [0]
                            f.write(f"{self.step}, {np.mean(self.episode_reward)}, {self.eps:.4f}, {np.mean(self.losses)}\n")
                        self.episode_reward = [0]
                        self.losses = []
        
    def run_updates(self):
        self.policy_net.train()
        
        for _ in range(self.cfg.train.grad_steps):
            replay_data = self.buffer.sample(self.cfg.train.batch_size)
            
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_tens = self.obs_to_tensor(replay_data.next_observations)
                next_q_values = self.target_net(next_tens)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.cfg.env.gamma * next_q_values
            
            # Get current Q-values estimates
            obs_tens = self.obs_to_tensor(replay_data.observations)
            current_q_values = self.policy_net(obs_tens)
            
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            self.losses.append(loss.item())
            
            # Optimize the policy
            self.optim.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.train.max_grad_norm)
            self.optim.step()
    
    def train(self):
        while self.step < self.cfg.train.max_steps:
            self.run_rollouts()
            
            if self.step > self.cfg.train.start_step:
                self.run_updates()
        

if __name__ == "__main__":
    config_path = "/home/jason/projects/Learning-RL/q_networks/configs/dqn_config.yaml"
    
    model = DQNTrainer(config_path)
    model.train()