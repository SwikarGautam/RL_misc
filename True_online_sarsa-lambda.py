import gym
import numpy as np
from tile_coding import tilings, features
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

actions = list(range(env.action_space.n))

num_tile = 16
num_tiling = 16

tiling = tilings([[-2.5, -3.5, -0.3, -4],
                  [2.5, 3.5, 0.3, 4]], num_tiling, num_tile)

weight = np.zeros(
    (num_tiling * (num_tile**env.observation_space.shape[0]), env.action_space.n))
# weight = np.zeros(((16*16*16*16*16), env.action_space.n))
z = np.zeros_like(weight)


def q(state, action, weights=weight):

    return np.sum(weights[features(tiling, state), action])


ep = 0.3
alpha = 1/48
gamma = 1
lambdha = 0.3
n_episode = 200

indexes = set()

for episode in range(n_episode):
    print('episode:', episode)
    obs = env.reset()
    act = max(actions, key=lambda x: q(obs, x)
              ) if np.random.uniform() > ep else env.action_space.sample()
    indexes = [[], []]
    z = np.zeros_like(weight)
    v_old = 0
    total_reward = 0

    while True:
        if episode >= n_episode-5:
            print(q(obs, act))
            env.render()
        n_obs, reward, done, _ = env.step(act)
        total_reward += reward

        index = features(tiling, obs)

        for i in index:
            occ = np.where(np.array(indexes[0]) == i)[0]
            flag = False
            for o in occ:
                if indexes[1][o] == act:
                    flag = True
            if not flag:
                indexes[0].append(i)
                indexes[1].append(act)

        delta = reward - q(obs, act)

        if not done:
            n_act = max(actions, key=lambda x: q(n_obs, x)
                        ) if np.random.uniform() > ep else env.action_space.sample()
            next_q = q(n_obs, n_act)
            delta += gamma*next_q

        z[tuple(indexes)] *= gamma*lambdha
        z[index, act] += (1-alpha*np.sum(z[index, act]))

        weight[tuple(indexes)] += alpha * \
            (delta+q(obs, act)-v_old)*z[tuple(indexes)]
        weight[index, act] -= alpha*(q(obs, act)-v_old)

        if done:
            print(total_reward)
            break

        obs = n_obs
        act = n_act
        v_old = next_q

env.close()
plt.plot(weight)
plt.show()
