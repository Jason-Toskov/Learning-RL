import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm import tqdm

from models import QNetwork
from buffers import RandomBuffer


class DQNTrainer:
    def __init__(self):
        self.vec_env = make_atari_env("BoxingNoFrameskip-v4")
        self.env = VecFrameStack(self.vec_env, n_stack=4)
        self.obs_shape = self.env.observation_space.shape
        self.obs_dtype = self.env.observation_space.dtype
        self.action_dim = 1
        self.action_len = self.env.action_space.n
        self.action_dtype = self.env.action_space.dtype
        self.gamma = 0.99
        self.frame_stack = 4
        self.last_obs = self.env.reset()
        
        
        self.step = 0
        self.max_steps = 10_000_000
        self.pbar = tqdm(range(self.max_steps))
        
        self.start_learning = 100_000
        
        self.eps = 1
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_steps_final = 1_000_000
        
        self.train_freq = 4
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.policy_net = QNetwork(self.frame_stack, self.env.action_space.n).to(self.device)
        self.target_net = QNetwork(self.frame_stack, self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update_interval = 1000
        self.batch_size = 32
        self.max_grad_norm = 10
        self.grad_steps = 1

        self.buffer = RandomBuffer(
            buffer_size = 100_000,
            obs_shape = self.obs_shape,
            obs_dtype = self.obs_dtype,
            action_dim = self.action_dim,
            act_dtype = self.action_dtype,
            device = self.device
        )
        
        self.episode_num = 0
        self.episode_reward = [0]
        self.losses = []
        
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
    def obs_to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs)
        obs = torch.permute(obs, (0,3,1,2))
        obs = (obs.float() / 255)
        return obs
        
    def sample_actions(self, random=False):
        if self.step < self.start_learning or random or np.random.rand() < self.eps:
            action = np.array([self.env.action_space.sample()])
        else:
            obs = self.obs_to_tensor(self.last_obs).to(self.device)
            with torch.no_grad():
                action = self.policy_net(obs).argmax(dim=1).reshape(-1).cpu().numpy()
        
        return action
    
    def end_step(self):
        if self.step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.eps > self.eps_end:
            self.eps = self.eps - ((self.eps_start - self.eps_end) / self.eps_steps_final)
        else:
            self.eps = self.eps_end
            
        self.step += 1
        
        self.pbar.update(1)
        
    def run_rollouts(self):
        self.policy_net.eval()
        for rollout_step in range(self.train_freq):
            
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
        
        for _ in range(self.grad_steps):
            replay_data = self.buffer.sample(self.batch_size)
            
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_tens = self.obs_to_tensor(replay_data.next_observations)
                next_q_values = self.target_net(next_tens)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
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
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optim.step()
    
    def train(self):
        while self.step < self.max_steps:
            self.run_rollouts()
            
            if self.step > self.start_learning:
                self.run_updates()
        

if __name__ == "__main__":
    
    
    model = DQNTrainer()
    model.train()