from DeepDenseNetwork.DeepDenseNetwork import DeepDenseNet
import numpy as np
import gym
from tile_coding import features, tilings
import copy
import pickle
import matplotlib.pyplot as plt
import time


env = gym.make('CartPole-v0')

baseline = True


NN = DeepDenseNet([env.observation_space.shape[0], 128,
                   env.action_space.n])
if baseline:
    tiling = tilings([[-2.5, -3.5, -0.3, -4],
                      [2.5, 3.5, 0.3, 4]], 16, 16)

    weight = np.zeros((16*16**4, 1))
    #counting = np.zeros_like(weight)

    def v(obs, weights=weight):
        return np.sum(weights[features(tiling, obs)])


alpha = 1e-3
alpha2 = 0.0001
gamma = 0.99
n_episodes = 500
batch_size = 10

for episode in range(n_episodes):

    obs = env.reset()
    sequence = []
    total_reward = 0
    counter = [0]*env.action_space.n
    ave_prob = np.zeros((1, env.action_space.n), dtype=np.float64)
    ave_logprob = 0
    time.sleep(0.35)
    while True:
        actions = NN.feed_forward(obs[np.newaxis, :])
        # print(actions)
        if episode > n_episodes-5:
            env.render()

        act = np.random.choice(np.arange(actions.size), p=np.squeeze(actions))

        n_obs, reward, done, _ = env.step(act)

        sequence.append([obs, act, actions[0][act], reward])

        ave_prob += actions
        ave_logprob += np.log(actions[0][act])
        counter[act] += 1
        total_reward += reward

        if done:
            out = NN.feed_forward(
                np.zeros((1, env.observation_space.shape[0])))
            print('episode:', episode, total_reward)
            break
        obs = n_obs

    ave_grad = np.zeros_like(actions[0])
    if episode % batch_size == 0:
        training_data = []

    ret = 0
    disc_return = []
    for i, s in enumerate(sequence[::-1]):
        reward = s[3]
        ret = reward + gamma*ret
        disc_return.append(ret)

    disc_return = np.array(disc_return)
    disc_return = (disc_return - np.mean(disc_return)) / \
        (np.std(disc_return)+1e-8)

    for i in range(len(sequence)):
        obs = sequence[i][0]
        act = sequence[i][1]
        ret = disc_return[-(i+1)]
        delta = ret
        if baseline:
            bs = v(obs)
            delta -= bs
            weight[features(tiling, obs)] += alpha2*ret
        delta *= gamma**i
        output = np.zeros((env.action_space.n, 1))

        output[act] = delta / (sequence[i][2])
        training_data.append([obs, output])

    ave_prob2 = np.zeros_like(ave_prob)

    if episode % batch_size == 0:
        NN.train_network(training_data, 1, -alpha, 200, noLoss=True)
        # training_data = []

    obs_list = [s[0] for s in sequence]
    action_list = [s[1] for s in sequence]
    prob2 = NN.feed_forward(np.array(obs_list))
    ave_prob2 = np.sum(prob2, axis=0)
    ave_logprob2 = np.sum(
        np.log(prob2[list(range(prob2.shape[0])), action_list]))

env.close()
