import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# REINFORCE算法网络
class MlpNet(nn.Module):

    # 网络输入8个量分别对应：b1 b2 b3 d1 d2 d3 pt sc
    def __init__(self):
        super(MlpNet, self).__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 17)

    # 网络输出out1是任务123边缘执行概率，out2是任务123带宽分配与边缘处理器速度分配的均值和方差，out3是公平性参数u的均值和方差
    def forward(self, state):
        out = self.linear1(state)
        out = nn.functional.relu(out)
        out = self.linear2(out)
        out = nn.functional.relu(out)
        out = self.linear3(out)
        out = nn.functional.relu(out)
        out = self.linear4(out)
        out1 = nn.functional.sigmoid(out[0:3])
        out2 = out[3:15]
        out3 = out[15:17]

        return out1, out2, out3


# 环境
class Env:
    def __init__(self):
        pass

    # 根据传入的状态和动作返回奖励
    def step(self, state, action):
        # 任务123的数据量
        b = [state[0].item(), state[1].item(), state[2].item()]
        # 任务123的cpu长度
        d = [state[3].item(), state[4].item(), state[5].item()]
        # 云服务器的cpu速度
        pt = state[6].item()
        # 边缘服务器和云服务器间的延迟
        sc = state[7].item()

        # 任务123是否在边缘执行，1代表是，0代表否
        x = [action[0].item(), action[1].item(), action[2].item()]
        # 任务123的带宽分配偏好
        w = [action[3].item(), action[4].item(), action[5].item()]
        # 任务123的边缘cpu速度分配偏好
        s = [action[6].item(), action[7].item(), action[8].item()]

        # 这里将公平性参数u的范围从R映射到(0,7)
        u = 8.31 * nn.functional.sigmoid(action[9]).item()
        u = math.e ** (u / 4) - 1

        # 这里保证u不能太小，否则可能导致数据过大训练失败
        if u < 1:
            u = 1

        # 将w和s的值从R映射到R+
        sum_w = 0
        sum_s = 0
        for i in range(3):
            w[i] = math.e ** (w[i] / u)
            s[i] = math.e ** (s[i] / u)
            sum_w += w[i]
            if x[i] == 1:
                sum_s += s[i]

        if sum_s == 0:
            sum_s = 1

        # 按照任务123的带宽分配偏好平分总带宽（这里总带宽设为7）
        # 按照任务123的边缘cpu速度分配偏好平分总边缘cpu速度（这里总边缘cpu速度设为10）
        for i in range(3):
            w[i] = w[i] / sum_w * 7
            s[i] = s[i] / sum_s * 10

        # 计算延迟
        delay = 0
        for i in range(3):
            delay += b[i] / w[i] + x[i] * d[i] / s[i] + (1 - x[i]) * (d[i] / sc + pt)

        # 如果任务选择在云执行，那么将不会分配边缘cpu速度
        if x[0] == 0:
            s[0] = 0
        if x[1] == 0:
            s[1] = 0
        if x[2] == 0:
            s[2] = 0

        # 本次决策结果
        policy = (x[0], x[1], x[2], w[0], w[1], w[2], s[0], s[1], s[2])

        # 本次奖励
        reward = -1 * torch.tensor(1.5 + delay * 0.05 + np.var(w))

        return reward, policy, delay, u

    # 随机产生一个状态 b1 b2 b3 d1 d2 d3 pt sc
    def generate_state(self):

        state = [3 * random.random() + 0.5, 3 * random.random() + 0.5, 3 * random.random() + 0.5,
                 10 * random.random() + 4, 10 * random.random() + 4, 10 * random.random() + 4,
                 abs(random.gauss(0.5, 1)), 5]

        return torch.tensor(state)


# 智能体
class Agent:
    def __init__(self, lr):
        self.model = MlpNet()
        # 优化器用SGD
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    # 根据当前状态和自己当前的策略选择动作，返回3个范围{0,1}的离散值，7个范围R的连续值
    def interacte(self, state):
        action = []

        # epsilon-greedy增加探索性，原REINFORCE算法并没有这一步，这里目前设置不进行随机探索，即epsilon = 0
        epsilon = 0
        if random.random() > epsilon:
            out1, out2, out3 = self.model(state)

            # out1[i]是在任务在边缘执行的概率，根据概率随机选择离散动作
            for i in range(3):
                if out1[i] > random.random():
                    action.append(1)
                else:
                    action.append(0)

            # 根据概率密度函数随机选择连续动作
            for i in range(6):
                # 这里取e的x次方防止方差为负
                action.append(np.random.normal(out2[2 * i].item(), math.sqrt(math.e ** out2[2 * i + 1].item())))

            # 根据概率密度函数随机选择连续动作u
            action.append(np.random.normal(out3[0].item(), math.sqrt(math.e ** out3[1].item())))

        else:
            action = [round(random.random()), round(random.random()), round(random.random()), 1 * random.random(),
                      1 * random.random(), 1 * random.random(), 1 * random.random(), 1 * random.random(),
                      1 * random.random()]

        return torch.tensor(action)

    # 高斯分布的概率密度函数，x不可反向传播，mean和variance可以反向传播
    def normal(self, x, mean, variance):
        a = (-1 * (x - mean).pow(2) / (2 * variance)).exp()
        b = 1 / (2 * math.pi * variance).sqrt()
        return a * b

    # 根据长度为1的episode进行学习
    def learn(self, state, action, reward):
        out1, out2, out3 = self.model(state)

        loss = torch.tensor(1, dtype=torch.float32)
        # loss累乘上3个离散动作概率和7个连续动作概率密度
        for i in range(3):
            loss *= out1[i] * action[i] + (torch.tensor(1, dtype=torch.float32) - out1[i]) * (
                    torch.tensor(1, dtype=torch.float32) - action[i])
        for i in range(6):
            loss *= self.normal(action[3 + i], out2[i * 2], math.e ** out2[i * 2 + 1])
        loss *= self.normal(action[9], out3[0], math.e ** out3[1])

        # 按照REINFORCE的公式进行更新
        loss = loss.log() * -1 * reward

        # 清空梯度缓存
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    # 保存模型
    def save_model(self, path):
        torch.save(self.model, path)


def main():
    # 每次迭代更新次数
    step = 200
    # 迭代次数
    epoch = 200

    # 学习率
    agent = Agent(0.000001)
    env = Env()

    sum = 0
    count = 0
    list_plot = []
    list_u = []

    for i in range(step * epoch):
        state = env.generate_state()
        action = agent.interacte(state)
        reward, policy, delay, u = env.step(state, action)
        agent.learn(state, action, reward)
        if delay < 100:
            sum += delay

        if i % step == 0 and i != 0:
            print("======================================================================")
            print("平均延迟：" + str(sum / step))
            list_plot.append(sum / step)
            print(policy)
            print("本次延迟：" + str(delay))
            print("本次奖励：" + str(reward))
            print("u=" + str(u))
            list_u.append(u)
            print("======================================================================")
            sum = 0

    plt.figure()
    plt.plot(list_plot, c="red")
    plt.show()

    agent.save_model(r"F:\PythonWorkspace\nonlinear_programming\PGModel.pkl")


if __name__ == "__main__":
    main()
