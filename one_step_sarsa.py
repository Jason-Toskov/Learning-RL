import gymnasium as gym
from gymnasium import Env
import numpy as np
from tqdm import tqdm


def sarsa(env_params: dict, gamma: float, alpha: float):
    env: Env = env_params["env"]
    nS = env_params["num_states"]
    nA = env_params["num_actions"]

    # N_0 = 100
    epsilon_decay = 0.99999
    eps = 1
    N_sa = np.zeros((*nS, nA))
    Q_sa = np.zeros((*nS, nA))
    policy = np.ones((*nS, nA)) / nA
    for ep in tqdm(range(env_params["num_episodes"])):
        s, info = env.reset()
        a = np.random.choice(nA, p=policy[s])
        eps *= epsilon_decay
        # Play an episode
        for step in range(env_params["max_iter"]):
            s_next, r, terminated, truncated, info = env.step(a)
            a_next = np.random.choice(nA, p=policy[s_next])

            # Update Q
            N_sa[s][a] += 1
            # alpha = 1 / N_sa[s][a]
            Q_sa[s][a] = Q_sa[s][a] + alpha * (
                r + gamma * Q_sa[s_next][a_next] - Q_sa[s][a]
            )

            # Update policy
            a_opt = np.argmax(Q_sa[s])
            # eps = N_0 / (N_0 + np.sum(N_sa[s]))
            policy[s] = eps / nA
            policy[s][a_opt] = 1 - eps + eps / nA

            s = s_next
            a = a_next

            if terminated or truncated:
                break

    return Q_sa, policy


if __name__ == "__main__":
    # env = gym.make("Blackjack-v1", sab=True)
    # env = gym.make("CliffWalking-v0")
    env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
    )

    gamma = 0.9
    alpha = 0.02
    # nS = [obs.n for obs in env.observation_space]
    nS = [env.observation_space.n]
    nA = env.action_space.n
    env_params = {
        "env": env,
        "num_states": nS,
        "num_actions": nA,
        "num_episodes": int(1e5),
        "max_iter": int(1e4),
    }
    Q_sa, policy = sarsa(env_params, gamma, alpha)

    # print(np.argmax(Q_sa, axis=-1)[10:-10, 1:, 0])
    # print(np.argmax(Q_sa, axis=-1)[10:-10, 1:, 1])

    # print((1e2 * value_func)[10:-10, 1:, 0].astype(int), "\n")
    # print((1e2 * value_func)[10:-10, 1:, 1].astype(int))

    print("Done!")
