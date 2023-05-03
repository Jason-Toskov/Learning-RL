import gymnasium as gym
from gymnasium import Env
import numpy as np
from tqdm import tqdm


def mc_policy_iteration(env_params: dict, gamma: int, first_visit: bool = True):
    env: Env = env_params["env"]
    nS = env_params["num_states"]
    nA = env_params["num_actions"]

    N_0 = 100

    state_act_count = np.zeros((*nS, nA))
    Q_sa = np.zeros((*nS, nA))
    policy = np.ones((*nS, nA)) / nA
    for ep in tqdm(range(env_params["num_episodes"])):
        obs, info = env.reset()

        # Play an episode
        ep_info = {"s": [], "a": [], "r": [], "s_next": []}
        for step in range(env_params["max_iter"]):
            # Policy is epsilon greedy from the updates
            action = np.random.choice(nA, p=policy[obs])
            ep_info["s"].append(obs)
            ep_info["a"].append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_info["r"].append(reward)
            ep_info["s_next"].append(obs)

            if terminated or truncated:
                break

        # MC policy eval
        G_t = 0
        for i, (s, a, r) in enumerate(
            list(zip(ep_info["s"], ep_info["a"], ep_info["r"]))[::-1]
        ):
            G_t = r + gamma * G_t
            # First visit to (s,a) pair is counted
            state_actions = list(zip(ep_info["s"], ep_info["a"]))[::-1][i + 1 :]
            if (s, a) not in state_actions or not first_visit:
                # Update action value
                state_act_count[s][a] += 1
                Q_sa[s][a] = Q_sa[s][a] + (G_t - Q_sa[s][a]) / state_act_count[s][a]

                # Update policy
                a_opt = np.argmax(Q_sa[s])
                eps = N_0 / (N_0 + np.sum(state_act_count[obs]))
                policy[s] = eps / nA
                policy[s][a_opt] = 1 - eps + eps / nA
            else:
                pass
                # print("repeat state")

    return Q_sa, policy


if __name__ == "__main__":
    # env = gym.make("Blackjack-v1", sab=True)
    env = gym.make("CliffWalking-v0")

    gamma = 1
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
    Q_sa, policy = mc_policy_iteration(env_params, gamma, first_visit=True)

    # print(np.argmax(Q_sa, axis=-1)[10:-10, 1:, 0])
    # print(np.argmax(Q_sa, axis=-1)[10:-10, 1:, 1])

    # print((1e2 * value_func)[10:-10, 1:, 0].astype(int), "\n")
    # print((1e2 * value_func)[10:-10, 1:, 1].astype(int))

    print("Done!")
