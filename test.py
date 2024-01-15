import os
import numpy as np
import paddle
import torch
from matplotlib import pyplot as plt
import paddle.nn.functional as F
from paddle import nn
from stable_baselines3.common.vec_env import VecNormalize
from algorithm import PPO
from env import RobotEnv
from model import Model
from agent import PPOAgent
from storage import ReplayMemory

# 玩多少次
TRAIN_EPISODE = 100
# 到达的学习的数据数
UPDATE_TIMESTEP = 1000
# 学习率
LR = 0.0001
# adm更新参数
BETAS = (0.9, 0.99)
# 折扣因子
GAMMA = 0.95
# 学习的次数
K_EPOCHS = 4
# ppo截断
EPS_CLIP = 0.2


class NatureCNN(nn.Layer):
    def __init__(self):
        super(NatureCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2D(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x


class OnnxablePolicy(nn.Layer):
    def __init__(self):
        super(OnnxablePolicy, self).__init__()

        self.extractor = NatureCNN()
        self.action_net = nn.Linear(512, 3)
        self.value_net = nn.Linear(512, 1)

    def forward(self, x):
        x = self.extractor(x)
        action = self.action_net(x)
        value = self.value_net(x)
        return action, value


def run_evaluate_episodes(agent, env, max_epi):
    for i in range(max_epi):
        timestep = 0
        episode_reward = []
        obs = env.reset()
        while True:
            timestep += 1
            obs = np.transpose(obs, [2, 0, 1])
            obs = paddle.to_tensor(obs, dtype='float32')
            obs = obs.unsqueeze(0)
            # 归一化
            # obs = torch.from_numpy(obs).float()
            # obs = obs.unsqueeze(0)
            # observation_space = env.observation_space
            # obs = preprocess_obs(obs, observation_space)
            # print(obs)
            # obs = obs.detach().numpy()
            # obs = paddle.to_tensor(obs, dtype='float32')
            # action = model.policy(obs)
            # probs = F.softmax(action)
            # action = paddle.argmax(probs, 1)
            action = agent.predict(obs)

            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            episode_reward.append(reward)
            if done:
                return info, episode_reward


# 创建环境
env = RobotEnv(True)
# 使用PARL框架创建agent
model = Model()
# weight_npy = np.load('conv/conv1.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv1.weight.set_value(weight_tensor)
# weight_npy = np.load('conv/conv1_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv1.bias.set_value(weight_tensor)
# weight_npy = np.load('conv/conv2.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv2.weight.set_value(weight_tensor)
# weight_npy = np.load('conv/conv2_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv2.bias.set_value(weight_tensor)
# weight_npy = np.load('conv/conv3.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv3.weight.set_value(weight_tensor)
# weight_npy = np.load('conv/conv3_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.conv3.bias.set_value(weight_tensor)
# weight_npy = np.load('conv/fc1.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc1.weight.set_value(weight_tensor.T)
# weight_npy = np.load('conv/fc1_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc1.bias.set_value(weight_tensor)
# weight_npy = np.load('conv/fc2.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc2.weight.set_value(weight_tensor.T)
# weight_npy = np.load('conv/fc2_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc2.bias.set_value(weight_tensor)
# weight_npy = np.load('conv/fc3.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc3.weight.set_value(weight_tensor.T)
# weight_npy = np.load('conv/fc3_bias.npy')
# weight_tensor = paddle.to_tensor(weight_npy, dtype='float32')
# model.fc3.bias.set_value(weight_tensor)
# print("yes")

ppo = PPO(model, LR, BETAS, GAMMA, K_EPOCHS, EPS_CLIP)
agent = PPOAgent(ppo, model)
rpm = ReplayMemory()
# 导入策略网络参数
PATH = 'train_log/model1.ckpt'
# agent.save('../ppo/train_log/model1.ckpt')

episode = 0
it = 0
while it <= 9450:
    if os.path.exists(PATH):
        agent.restore(PATH)

        is_coll = 0
        episode = 0
        while episode < TRAIN_EPISODE:
            # info, episode_reward = run_evaluate_episodes(agent, env, TRAIN_EPISODE, model)
            info, episode_reward = run_evaluate_episodes(agent, env, TRAIN_EPISODE)

            if info["iscoll"]:
                is_coll += 1
            episode += 1
            # plt.plot(episode_reward)
            # plt.show()
        print("it : {}   is_coll: {}".format(it, is_coll))
    it += 50
    PATH = 'train_log/model{}.ckpt'.format(it)