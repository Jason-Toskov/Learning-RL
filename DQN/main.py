from collections import deque
import random

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn as nn
from tqdm import tqdm

from enums import EnvType
from models import QNetwork


# @dataclass
# class Transition:
#     state: torch.Tensor
#     action: int
#     reward: float
#     next_state: torch.Tensor


class AtariState:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.reset()

    def reset(self):
        self._buffer = deque(maxlen=self.n_frames)

    def add_frame(self, frame):
        f_img = torch.tensor(frame / 255).unsqueeze(0).unsqueeze(0)
        f_ds = nn.functional.interpolate(f_img, (110, 84))
        f_out = T.functional.center_crop(f_ds, 84)
        self._buffer.append(f_out.squeeze())

    @property
    def enough_frames(self):
        return len(self._buffer) == self.n_frames

    @property
    def state(self) -> torch.Tensor:
        if not self.enough_frames:
            raise ValueError(f"Buffer only has {len(self._buffer)} frames")
        else:
            t = torch.stack(tuple(self._buffer)).to(torch.float32)
            return t


class DQN:
    def __init__(self):
        print("Env creation")
        self.env_type = EnvType.pong
        self.env = gym.make(self.env_type, obs_type="grayscale")
        self.state_buffer = AtariState(4)

        self.step = 0
        self.max_steps = int(1e7)

        self.eps = 1
        self.gamma = 0.99
        self.n_frames_in_state = 4

        print("Model creation")
        # Model
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.q_function = QNetwork(self.n_frames_in_state, self.env.action_space.n)
        self.q_function = self.q_function.to(self.device)
        self.batch_size = 32

        self.loss = nn.MSELoss()
        self.optim = torch.optim.RMSprop(
            self.q_function.parameters(), lr=2.5e-4, momentum=0.95
        )

        # Evaluation
        self.eval_frequency = 20000
        self.eval_eps = 0.05
        self.eval_horizon = 2000

        print("Buffer creation")
        self.buffer_length = 10000
        self.replay_buffer = deque(maxlen=self.buffer_length)
        self.init_buffer_size = 10000
        print("Populating buffer")
        self.pre_populate_buffer()

    def reset_env(self):
        state, info = self.env.reset()
        self.state_buffer.reset()
        self.state_buffer.add_frame(state)
        while not self.state_buffer.enough_frames:
            action = np.random.randint(self.env.action_space.n)
            new_frame, _, _, _, _ = self.env.step(action)
            self.state_buffer.add_frame(new_frame)

    def pre_populate_buffer(self):
        self.reset_env()
        with tqdm(total=self.init_buffer_size) as pbar:
            while len(self.replay_buffer) < self.init_buffer_size:
                action = np.random.randint(self.env.action_space.n)
                self.env_step(action)
                pbar.update(1)

    def sample_action(self) -> int:
        # Greedy if succeeds, else random
        if np.random.uniform() > self.eps:
            state_in = self.state_buffer.state.unsqueeze(0).to(self.device)
            action_prob = self.q_function(state_in).squeeze(0)
            action = torch.argmax(action_prob).cpu().item()
        else:
            action = np.random.randint(self.env.action_space.n)
        assert isinstance(action, int)
        return action

    def reward_filter(self, reward):
        if reward == 0:
            return reward
        elif reward > 0:
            return 1
        elif reward < 0:
            return -1
        else:
            raise ValueError("????????")

    def env_step(self, action=None):
        old_state = self.state_buffer.state
        if action is None:
            action = self.sample_action()

        new_frame, reward, terminated, truncated, info = self.env.step(action)
        self.state_buffer.add_frame(new_frame)

        reward_out = self.reward_filter(reward)
        next_state = self.state_buffer.state
        episode_live = not (terminated or truncated)

        # new_transition = Transition(old_state, action, reward_out, next_state)
        new_transition = [old_state, action, reward_out, next_state, episode_live]
        self.replay_buffer.append(new_transition)

        if not episode_live:
            self.reset_env()

    def update_q_func(self):
        # Sample batch from buffer
        batch_list = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, not_terminal = list(zip(*batch_list))

        # Get predicted Q values
        s_t = torch.stack(states).to(self.device)  # Bx4x84x84
        a_t = torch.tensor(actions, device=self.device).unsqueeze(-1)  # Bx1
        preds = self.q_function(s_t).gather(1, a_t).squeeze(-1)

        # Get TD targets
        r_t = torch.tensor(rewards, device=self.device)  # Bx1
        end_mask = torch.tensor(not_terminal, dtype=int, device=self.device)
        with torch.no_grad():
            s_next_t = torch.stack(next_states).to(self.device)
            next_q_vals, _ = self.q_function(s_next_t).max(axis=1)
            targets = r_t + self.gamma * end_mask * next_q_vals

        # Do an update step
        error = self.loss(targets, preds)
        self.optim.zero_grad()
        error.backward()
        self.optim.step()

    def decay_epsilon(self):
        # Linear decay for 1M steps from 1 to 0.1
        if self.eps > 0.1:
            self.eps = self.eps - (1 - 0.1) / 1e6

    def eval_average_reward(self):
        rewards_list = []
        eval_env = gym.make(self.env_type, obs_type="grayscale")

        def get_action_eval(env, eps, buffer):
            if np.random.uniform() > eps:
                state_in = buffer.state.unsqueeze(0).to(self.device)
                action_prob = self.q_function(state_in).squeeze(0)
                action = torch.argmax(action_prob).cpu().item()
            else:
                action = np.random.randint(env.action_space.n)
            assert isinstance(action, int)
            return action

        def reset_eval_env(env, rewards_list):
            state, info = env.reset()
            buffer = AtariState(4)
            buffer.reset()
            buffer.add_frame(state)
            while not buffer.enough_frames:
                action = np.random.randint(env.action_space.n)
                new_frame, _, _, _, _ = env.step(action)
                buffer.add_frame(new_frame)
            rewards_list.append(0)
            return env, buffer, rewards_list

        eval_env, eval_buffer, rewards_list = reset_eval_env(eval_env, rewards_list)
        for eval_step in range(self.eval_horizon):
            action = get_action_eval(eval_env, self.eval_eps, eval_buffer)
            new_frame, reward, terminated, truncated, info = self.env.step(action)
            eval_buffer.add_frame(new_frame)
            rewards_list[-1] += reward

            if terminated or truncated:
                eval_env, eval_buffer, rewards_list = reset_eval_env(
                    eval_env, rewards_list
                )

        # Our metric is the mean reward per episode
        return sum(rewards_list[:-1]) / len(rewards_list[:-1])

    def plot_rewards(self, rewards):
        plt.plot(range(len(rewards)), rewards)
        plt.grid(True)
        plt.xlabel("Training Epochs")
        plt.ylabel("Average Reward per Episode")
        plt.title(f"Average Reward on {self.env_type.name.upper()}")
        plt.savefig("./rewards_plot.png")
        plt.clf()

    def train(self):
        print("Begin training")
        self.reset_env()
        average_rewards_list = []

        for step in tqdm(range(self.max_steps)):
            # Perform one step
            self.env_step()

            # Update Q network
            self.update_q_func()

            # Decay epsilon
            self.decay_epsilon()

            if step % self.eval_frequency == 0:
                avg_reward = self.eval_average_reward()
                print(f"Average reward at step {step} is {avg_reward}")
                average_rewards_list.append(avg_reward)
                # self.plot_rewards(average_rewards_list)


if __name__ == "__main__":
    model = DQN()
    model.train()
