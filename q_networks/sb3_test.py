from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

if __name__ == "__main__":
    vec_env = make_atari_env("PongNoFrameskip-v4")
    env = VecFrameStack(vec_env, n_stack=4)

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=100_000,
        learning_starts=100_000,
        target_update_interval=1000,
        verbose=1,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        learning_rate=1e-4,
    )

    model.learn(total_timesteps=int(1e7), progress_bar=True)