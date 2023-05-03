import gymnasium as gym
import numpy as np

from policy_iteration import policy_improvement


def value_iteration(env, goal_S, gamma, min_error, max_iter):
    error = 1e9
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n

    val_func = np.zeros(nS)
    i = 0
    while error > min_error and i < max_iter:
        val_func_new = np.zeros(nS)
        for state in range(nS):
            # Terminal state V(s) = 0 by definition
            if state in goal_S:
                val_func_new[state] = 0
                continue
            Q_a = np.zeros(nA)
            for action in range(nA):
                action_accum = 0
                for trans_pr, next_state, reward, _ in P[state][action]:
                    action_accum += trans_pr * (reward + gamma * val_func[next_state])
                Q_a[action] = action_accum
            val_func_new[state] = np.max(Q_a)
        error = np.sum(np.abs(val_func_new - val_func))
        i += 1
        val_func = val_func_new

    print(f"Final error: {error}")
    print(f"Iterations: {i}")
    return val_func


if __name__ == "__main__":
    env_base = gym.make(
        "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array"
    )
    env = gym.wrappers.RecordVideo(
        env_base, "./outputs", episode_trigger=lambda x: x % 5 == 0
    )
    observation, info = env.reset()

    nS = env.observation_space.n
    nA = env.action_space.n
    goal_S = np.where((env.desc == b"G").flatten())[0]

    # Value iteration
    gamma = 0.9
    i = 0
    opt_val_func = value_iteration(env, goal_S, gamma, 1e-3, 1000)
    opt_policy = policy_improvement(opt_val_func, env, gamma)

    print((100 * opt_val_func).reshape(8, 8).astype(int))
    print(np.argmax(opt_policy, axis=1).reshape(8, 8))

    # Play with learned policy
    for ep in range(1):
        for _ in range(1000):
            action = np.argmax(opt_policy[observation])
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        observation, info = env.reset()

    env.close()
