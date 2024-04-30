import numpy as np
import gym
from tile_coding import features, tilings
import convo_net.ConvoNet as cn


env = gym.make('MountainCar-v0')


actor = cn.ConvoNet(
    (env.observation_space.shape[0], 1, 1), cn.NoLoss())

actor.add(
    cn.Dense(env.observation_space.shape[0], 128, cn.Tanh()))
actor.add(cn.Dense(128, env.action_space.n, cn.Softmax()))

critic = cn.ConvoNet((env.observation_space.shape[0], 1, 1),
                     cn.NoLoss())

critic.add(
    cn.Dense(env.observation_space.shape[0], 128, cn.Tanh()))
critic.add(cn.Dense(128, 1, cn.LeakyReLu(1)))

alpha1 = 4e-4
alpha2 = 0.009
gamma = 0.99
nepisodes = 1000
mbsize = 10

for episode in range(nepisodes):
    actor_train_data = []
    critic_train_data = []
    sequence = []
    obs = env.reset()
    i = 1
    I = 1
    while True:
        actions = actor.predict(obs[np.newaxis, :])
        act = np.random.choice(np.arange(actions.size), p=np.squeeze(actions))
        if episode > nepisodes-10:
            env.render()
        n_obs, reward, done, _ = env.step(act)
        value = critic.predict(obs[np.newaxis, :])
        delta = reward - value + (1-done)*gamma * \
            critic.predict(n_obs[np.newaxis, :])

        target = np.zeros((env.action_space.n,))
        target[act] = I*delta/np.squeeze(actions)[act]
        actor_train_data.append([obs, target])
        critic_train_data.append([obs, delta[0]])
        if i % mbsize == 0:
            actor.train(actor_train_data, 1, -alpha1, mbsize)

            critic.train(critic_train_data,
                         1, -alpha2, mbsize)
            actor_train_data = []
            critic_train_data = []
            if done:
                print(episode, i)
                break
        obs = n_obs
        if done:
            actor.train(actor_train_data, 1, -alpha1, mbsize)
            critic.train(critic_train_data,
                         1, -alpha2, mbsize)
            print(episode, i)
            break
        i += 1
        I *= gamma


env.close()
