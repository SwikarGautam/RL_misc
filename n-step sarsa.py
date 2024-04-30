import gym
from tile_coding import tilings, features
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

actions = list(range(env.action_space.n))

num_tile = 16
num_tilings = 16

tiling = tilings([[-2.5, -3.5, -0.3, -4],
                  [2.5, 3.5, 0.3, 4]], num_tilings, num_tile)

weight = np.zeros(
    (num_tilings * (num_tile**env.observation_space.shape[0]), env.action_space.n))


def q(state, action, weight=weight):
    indexes = features(tiling, state)
    return np.sum(weight[np.array(indexes), action])


ep = 0.2  # 0.5        #0.1
alpha = 1/(16*2)  # 10  #2
alpha_decay = 1  # 1
ep_decay = 0  # 0.98 #1
gamma = 1
n = 4
temp = [0, 1, 2]  # temporary list
e = 100


length2 = []
epsilon = []
alphas = []
ave = 0


for episode in range(e):

    alpha *= alpha_decay
    if episode > 150:  # 250
        ep *= ep_decay

    print("episode:", episode)

    n_obs = env.reset()
    temp[0] = n_obs
    n_act = max(actions, key=lambda x: q(n_obs, x)
                ) if np.random.uniform() > ep else env.action_space.sample()
    temp[1] = n_act
    memory = []  # memory of all required states visited

    t = 1
    T = 0
    done = False

    while True:
        if not done:
            # print(n_obs)
            # print(features(tiling, n_obs))
            if episode >= e-10:
                env.render()
                print(q(n_obs, n_act))
            n_obs, reward, done, _ = env.step(n_act)
            if done:
                T = t
                print('len:', t-1)
                ave = 0.9*ave + 0.1*(t-1)
                length2.append(ave)
                epsilon.append(ep)
                alphas.append(alpha)

            temp[2] = reward
            memory.append([*temp])

            n_act = max(actions, key=lambda x: q(n_obs, x)
                        ) if np.random.uniform() > ep else env.action_space.sample()

            temp[0] = n_obs
            temp[1] = n_act

        if t - n >= 0:
            ret = sum((gamma**i)*x[2] for i, x in enumerate(memory))

            if not done:
                ret += (gamma**n)*q(n_obs, n_act)

            weight[features(tiling, memory[0][0]), memory[0][1]] += alpha * \
                (ret - q(memory[0][0], memory[0][1]))

            if done and t-n >= T-1:
                break

            del memory[0]
        t += 1

env.close()


# plt.plot(weight)
# plt.show()
plt.plot(length2)
plt.plot(np.array(epsilon)*(max(length2)/max(epsilon)))
plt.plot(np.array(alphas)*(max(length2)/max(alphas)))
plt.show()
