import gym
import convo_net.ConvoNet as cn
import numpy as np

env = gym.make('CartPole-v0')

actions = list(range(env.action_space.n))
NN = cn.ConvoNet((env.observation_space.shape[0], 1, 1), cn.NoLoss())

NN.add(cn.Dense(env.observation_space.shape[0], 128, cn.Tanh()))
NN.add(cn.Dense(128, env.action_space.n, cn.LeakyReLu(1)))


ep = 0.1
ep_decay = 0.98
alpha = 0.008
gamma = 0.99

n_epi = 250

mbsize = 10


def q(state, action):
    action_values = NN.predict(state[np.newaxis, :])
    return action_values[0][action]


for epi in range(n_epi):
    ep *= ep_decay
    state = env.reset()
    act = max(actions, key=lambda x: q(state, x)
              ) if np.random.uniform() > ep else env.action_space.sample()
    t_reward = 0
    i = 1
    training_data = []
    while True:
        if epi > n_epi-5:
            env.render()
        next_state, reward, done, _ = env.step(act)
        next_act = max(actions, key=lambda x: q(next_state, x)
                       ) if np.random.uniform() > ep else env.action_space.sample()
        delta = reward + (1-done)*gamma*q(next_state, next_act) - q(state, act)
        target = np.zeros((env.action_space.n, 1))
        target[act] = delta
        #av1 = NN.predict(state[np.newaxis, :])
        training_data.append([state, target])
        if i % mbsize == 0:
            # print(training_data[0])
            NN.train(training_data, 1, -alpha, mbsize)
            training_data = []
        #av2 = NN.predict(state[np.newaxis, :])
        #print(av1, av2, av2-av1, act, delta)
        state = next_state
        act = next_act
        t_reward += reward
        if done:
            NN.train(training_data, 1, -alpha, mbsize)
            print(t_reward)
            break
        i += 1
env.close()
