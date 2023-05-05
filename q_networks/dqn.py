import yaml

import numpy as np
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from q_networks import ROOT
from q_networks.utils.buffers import RandomBuffer
from q_networks.utils.helpers import obs_to_tensor
from q_networks.utils.models import QNetwork
from q_networks.configs.params import DQNParams
from q_networks.eval import Evaluator


class DQNTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = DQNParams.parse_obj(yaml.safe_load(f))
            
        self.tb = SummaryWriter(ROOT / self.cfg.log.path / self.cfg.log.run)
        # self.tb.add_hparams(hparam_dict=self.cfg,run_name=self.cfg.log.run)
        
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
        self.pbar.set_description("train step")
        
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
        
        self.episode_reward = 0
        self.losses = []
        
        self.optim = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.cfg.train.lr,
        )
        
        self.evaluator = Evaluator(self.cfg, self.tb)
        
    def sample_actions(self, random=False):
        if self.step < self.cfg.train.start_step or random or np.random.rand() < self.eps:
            action = np.array([self.env.action_space.sample()])
        else:
            obs = obs_to_tensor(self.last_obs).to(self.device)
            with torch.no_grad():
                action = self.policy_net(obs).argmax(dim=1).reshape(-1).cpu().numpy()
        
        return action
    
    def end_step(self):
        self.tb.add_scalar("train/epsilon", self.eps, self.step)
        self.tb.add_scalar("train/episode", self.episode_num, self.step)
        
        if self.step % self.cfg.train.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.eps = self.cfg.eps.schedule(
            self.step, 
            self.cfg.eps.start, 
            self.cfg.eps.end, 
            self.cfg.eps.steps
        )
        
        if self.step % self.cfg.eval.frequency == 0:
            self.evaluator.run_evaluation(self.policy_net, self.step)
        
        self.step += 1
        self.pbar.update(1)
        
    def end_episode(self):
        self.tb.add_scalar("train/episode_reward", self.episode_reward, self.step)
        if len(self.losses) == 0:
            self.losses = [0]
        self.tb.add_scalar("train/mean_episode_loss", np.mean(self.losses), self.step)
        
        self.episode_reward = 0
        self.losses = []
        self.episode_num += 1
        self.last_obs = self.env.reset()

    def run_rollouts(self):
        self.policy_net.eval()
        for rollout_step in range(self.cfg.train.update_freq):
            
            action = self.sample_actions()
            new_obs, rewards, dones, infos = self.env.step(action)
            self.buffer.add(self.last_obs, new_obs, action, rewards, dones, infos)
            
            self.episode_reward += rewards[0]
            self.last_obs = new_obs
            self.end_step()
            
            for idx, done in enumerate(dones):
                if done:
                    self.end_episode()
        
    def run_updates(self):
        self.policy_net.train()
        for _ in range(self.cfg.train.grad_steps):
            replay_data = self.buffer.sample(self.cfg.train.batch_size)
            
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_tens = obs_to_tensor(replay_data.next_observations)
                target_q_next = self.target_net(next_tens)
                if self.cfg.net.ddqn:
                    # Choose action from policy net
                    policy_q_next = self.policy_net(next_tens)
                    policy_actions = policy_q_next.argmax(dim=1).reshape(-1,1)
                    # Q values are the Q from the target given the policy action
                    next_q_values = torch.gather(target_q_next, dim=1, index=policy_actions.long())
                else:
                    # Follow greedy policy: use the one with the highest value
                    next_q_values = target_q_next.max(dim=1)[0].reshape(-1, 1)
                    
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.cfg.env.gamma * next_q_values
            
            # Get current Q-values estimates
            obs_tens = obs_to_tensor(replay_data.observations)
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