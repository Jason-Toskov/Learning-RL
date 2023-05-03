import gymnasium as gym
import numpy as np


# Policy eval
def policy_evaluation(policy, val_func, env, goal_S, gamma, min_error, max_iter):
    error = 1e9
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n

    i = 0
    while error > min_error and i < max_iter:
        val_func_new = np.zeros(nS)
        for state in range(nS):
            # Terminal state V(s) = 0 by definition
            if state in goal_S:
                val_func_new[state] = 0
                continue
            state_val_accum = 0
            for action in range(nA):
                a_given_s_pr = policy[state][action]  # state trans pr
                action_accum = 0
                for trans_pr, next_state, reward, _ in P[state][action]:
                    # This formula feels a bit weird and different from the
                    # bellman eqns (because we take the next step reqard, not
                    # the reward from the current state), but I think its all
                    # that can be done in a gym env like this
                    action_accum += trans_pr * (reward + gamma * val_func[next_state])
                state_val_accum += a_given_s_pr * action_accum
            val_func_new[state] = state_val_accum
        error = np.sum(np.abs(val_func_new - val_func))
        i += 1
        val_func = val_func_new
    return val_func


def policy_improvement(val_func, env, gamma):
    P = env.P
    nS = env.observation_space.n
    nA = env.action_space.n

    # Policy improvement
    policy = np.zeros((nS, nA))
    action_vals = np.zeros((nS, nA))
    # Get all the action values for each state
    for state in range(nS):
        for action in range(nA):
            for trans_pr, next_state, reward, _ in P[state][action]:
                action_vals[state][action] += trans_pr * (
                    reward + gamma * val_func[next_state]
                )
        policy[state][np.argmax(action_vals[state])] = 1
    return policy


if __name__ == "__main__":
    env_base = gym.make(
        "FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode="rgb_array"
    )
    # env_base = gym.make("CliffWalking-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env_base, "./outputs", episode_trigger=lambda x: x % 5 == 0
    )
    # env = env_base

    observation, info = env.reset()

    nS = env.observation_space.n
    nA = env.action_space.n
    policy = np.ones((nS, nA)) / nA
    val_func = np.zeros(nS)

    # goal_S = np.array([47])
    goal_S = np.where((env.desc == b"G").flatten())[0]

    # Policy iteration
    gamma = 0.99
    policy_error = 1e9
    i = 0
    while i < 20 and policy_error > 0:
        val_func = policy_evaluation(
            policy, np.zeros(nS), env, goal_S, gamma, 1e-3, 1000
        )
        policy_new = policy_improvement(val_func, env, gamma)
        policy_error = np.sum(np.abs(policy - policy_new))
        policy = policy_new
        i += 1

    print(i)
    print(policy_error)

    print((100 * val_func).reshape(8, 8).astype(int))
    print(np.argmax(policy, axis=1).reshape(8, 8))
    # print((10 * val_func).reshape(4, 12).astype(int))
    # print(np.argmax(policy, axis=1).reshape(4, 12))

    # Play with learned policy
    for ep in range(1):
        for _ in range(1000):
            # action = env.action_space.sample()
            action = np.argmax(policy[observation])
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        observation, info = env.reset()

    env.close()
