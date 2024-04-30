from tile_coding import tilings, features
import gym
import numpy as np


env = gym.make('MountainCar-v0')

actions = list(range(env.action_space.n))

# print(env.action_space.n, [
#       env.observation_space.low, env.observation_space.high])

# tiling = tilings([env.observation_space.low,
#                   env.observation_space.high], num_tilings=32, num_tiles=8)
tiling = tilings([env.observation_space.low,
                  env.observation_space.high], 8, 4)

# tiling = tilings([env.observation_space.low,
#                   env.observation_space.high], 32, 4)

weight = np.zeros((4*4*8, env.action_space.n))


def q(state, action, weights=weight):
    indexes = features(tiling, state)
    return np.sum(weights[np.array(indexes), action])


ep = 0.05
alpha = 1/(160)
gamma = 0.99

for episode in range(200):
    print(episode)
    state = env.reset()
    action = max(actions, key=lambda x: q(state, x, weight)
                 ) if np.random.uniform() > ep else env.action_space.sample()

    while True:
        if episode > 195:
            env.render()

        next_state, reward, done, _ = env.step(action)

        if done:
            weight[np.array(features(tiling, state)), action] += alpha * \
                (reward - q(state, action, weight))
            break

        next_action = max(actions, key=lambda x: q(state, x)
                          ) if np.random.uniform() > ep else env.action_space.sample()

        weight[np.array(features(tiling, state)), action] += alpha*(reward + gamma*q(next_state, next_action) -
                                                                    q(state, action))

        action = next_action
        state = next_state

env.close()
