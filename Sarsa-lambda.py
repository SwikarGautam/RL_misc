import gym
import numpy as np
from tile_coding import features, tilings
import matplotlib.pyplot as plt


env = gym.make('MountainCar-v0')

actions = list(range(env.action_space.n))


current = [[1, 1, 1, 1, 6, 11], [-1, -1, -1, -1, -6, -11]]


def compare(obs, current=current):
    for i in range(len(obs)):
        if obs[i] > current[0][i]:
            current[0][i] = obs[i]

    for i in range(len(obs)):
        if obs[i] < current[1][i]:
            current[1][i] = obs[i]


# tiling = tilings([[1, 1, 1, 1, 6, 12], [-1, -1, -1, -1, -6, -12]][::-1], 32, [
#     4, 4, 4, 4, 4, 4])
tiling = tilings([env.observation_space.low, env.observation_space.high], 8,
                 4)


# weight = np.zeros((8*4*4, len(actions)))
weight = np.zeros((8*4**4, len(actions)))


def q(state, action, weights=weight):

    return np.sum(weights[features(tiling, state), action])


ep = 0
alpha = 1/(32*2)
gamma = 0.97
lambdha = 0.3
n_episode = 200


for episode in range(n_episode):
    print("episode:", episode)
    z = np.zeros_like(weight)
    indexes = [[], []]
    obs = env.reset()
    act = max(actions, key=lambda x: q(obs, x)
              ) if np.random.uniform() > ep else env.action_space.sample()
    # compare(obs)
    t_reward = 0
    if episode == n_episode-5:
        plt.plot(weight)
        plt.show()
    while True:

        if episode >= n_episode-5:
            env.render()

        n_obs, reward, done, _ = env.step(act)
        t_reward += reward
        # compare(obs)
        index = features(tiling, obs)

        for i in index:
            occ = np.where(np.array(indexes[0]) == i)[0]
            flag = False
            if len(occ) > 3:
                print('error')
            for o in occ:
                if indexes[1][o] == act:
                    flag = True
            if not flag:
                indexes[0].append(i)
                indexes[1].append(act)

        z[index, act] += 1

        delta = reward - q(obs, act)

        if not done:
            n_act = max(actions, key=lambda x: q(n_obs, x)) if np.random.uniform(
            ) > ep else env.action_space.sample()
            delta += gamma*q(n_obs, n_act)

        weight[tuple(indexes)] += alpha*delta*z[tuple(indexes)]

        z[tuple(indexes)] *= gamma*lambdha

        if done:
            print(t_reward)
            break
        obs = n_obs
        act = n_act

env.close()


# print(current)
#print('length of indexes', len(indexes))
