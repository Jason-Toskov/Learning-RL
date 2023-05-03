import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
from tqdm import tqdm


def mc_policy_evaluation(
    policy: np.ndarray, env_params: dict, gamma: float, first_visit: bool = True
):
    env: Env = env_params["env"]
    nS = env_params["num_states"]
    nA = env_params["num_actions"]

    state_counter = np.zeros(nS)
    V_s = np.zeros(nS)
    n_complete_ep = 0
    pbar = tqdm(range(env_params["num_episodes"]))
    for ep in pbar:
        obs, info = env.reset()
        # Play an episode
        ep_info = {"s": [], "a": [], "r": [], "s_next": []}
        for step in range(env_params["max_iter"]):
            action = np.random.choice(nA, p=policy[obs])
            ep_info["s"].append(obs)
            ep_info["a"].append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_info["r"].append(reward)
            ep_info["s_next"].append(obs)

            if terminated or truncated:
                n_complete_ep += 1
                break
        pbar.set_description(f"% complete eps = {int(100 * n_complete_ep / (ep+1))}")

        # Perform an MC update on the episode
        G_t = 0
        for i, (s, r) in enumerate(list(zip(ep_info["s"], ep_info["r"]))[::-1]):
            G_t = r + gamma * G_t
            # TODO: check this reverses properly
            # Should be fixed but still...
            if s not in ep_info["s"][::-1][i + 1 :] or not first_visit:
                state_counter[s] += 1
                V_s[s] = V_s[s] + (G_t - V_s[s]) / state_counter[s]

    return V_s


def state_tuple_to_int(state: tuple[int], obs_space: tuple[Discrete]):
    state_int = 0
    state_multiplier = 1
    for s, o in zip(state, obs_space):
        state_int = state_int + state_multiplier * s
        state_multiplier *= o.n
    return state_int


if __name__ == "__main__":
    # env = gym.make(
    #     "FrozenLake-v1",
    #     map_name="8x8",
    #     is_slippery=False,
    #     render_mode="rgb_array",
    # )
    # env = gym.wrappers.RecordVideo(
    #     env_base,
    #     "./outputs",
    #     episode_trigger=lambda x: x % 5 == 0,
    # )
    # env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    env = gym.make("Blackjack-v1")
    obs, info = env.reset()

    gamma = 0.9
    nS = [obs.n for obs in env.observation_space]
    nA = env.action_space.n
    # policy = np.ones((*nS, nA)) / nA # Random
    policy = np.ones((*nS, nA))
    # Stick on 20 and 21 only
    policy[:, :, :, 0] = 0
    policy[20:, :, :, 1] = 0
    policy[20:, :, :, 0] = 1
    env_params = {
        "env": env,
        "num_states": nS,
        "num_actions": nA,
        "num_episodes": int(1e5),
        "max_iter": int(1e4),
    }
    value_func = mc_policy_evaluation(policy, env_params, gamma)

    print((1e2 * value_func)[10:-10, 1:, 0].astype(int), "\n")
    print((1e2 * value_func)[10:-10, 1:, 1].astype(int))

    print("Done!")
