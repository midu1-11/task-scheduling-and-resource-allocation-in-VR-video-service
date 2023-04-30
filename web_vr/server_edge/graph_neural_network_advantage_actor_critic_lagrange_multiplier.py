# 参考 https://blog.csdn.net/Er_Studying_Bai/article/details/128462002
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


# DQN的第一个trick，记录所有的四元组
class ReplayMemory():
    def __init__(self):
        self.memory = []

    # 存储单个四元组，其中state和next_state是两个元组，action和reward是两个值
    def push(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    # 随机采样一个batch_size大小的batch
    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = [], [], [], []
        for data in batch_data:
            batch_state.append([data[0][0][i][0] for i in range(len(data[0][0]))] + [data[0][0][i][1]
                                                                                     for i in range(len(data[0][0]))] +
                               data[0][1])
            batch_action.append(data[1])
            batch_reward.append(data[2])
            batch_next_state.append([data[3][0][i][0] for i in range(len(data[3][0]))] + [data[3][0][i][1]
                                                                                          for i in
                                                                                          range(len(data[3][0]))] +
                                    data[3][1])
        return torch.tensor(batch_state), torch.unsqueeze(torch.tensor(batch_action), 1), torch.unsqueeze(
            torch.tensor(batch_reward), 1), torch.tensor(
            batch_next_state)

    # 释放记录
    def free(self):
        self.memory = []


# Q网络，输入维度2 * n + 2，包括b1...bn,d1...dn,pt,sc，输出维度2的n次方对应所有动作选择
class QNet(nn.Module):
    def __init__(self, n):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(2 * n + 2, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, pow(2, n))

    def forward(self, state):
        out = self.linear1(state)
        out = nn.functional.relu(out)
        out = self.linear2(out)
        out = nn.functional.relu(out)
        out = self.linear3(out)
        out = nn.functional.relu(out)
        out = self.linear4(out)

        return out


class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        self.embed = nn.Linear(2, 32)
        self.W = nn.Linear(66, 128)
        self.V = nn.Linear(128, 1)

    # 图嵌入
    def forward(self, state):
        # 节点标准化
        x = self.normalization(state[0])
        pt_sc = torch.tensor(state[1])

        # h为节点嵌入
        h = torch.empty(0, 32)
        s = torch.empty(0, 1)
        hg = 0
        for i in range(x.shape[0]):
            x_embed = self.embed(x[i])
            h = torch.cat([h, x_embed.unsqueeze(0)], 0)
            hg += x_embed

        # hg为图信息
        hg = hg / x.shape[0]
        hc = torch.cat([hg, pt_sc], 0)

        # s为注意力打分
        for i in range(x.shape[0]):
            s = torch.cat([s, self.V(nn.functional.tanh(self.W(torch.cat([h[i], hc], 0)))).unsqueeze(0)], 0)

        # prob为任务选择在边缘服务器执行的概率
        prob = nn.functional.sigmoid(s)

        return prob

    # 节点标准化，将(b,d)任务映射到01图上
    def normalization(self, b_d):
        b_max = max([i[0] for i in b_d])
        b_min = min([i[0] for i in b_d])
        d_max = max([i[1] for i in b_d])
        d_min = min([i[1] for i in b_d])

        x = []
        for i in range(len(b_d)):
            x.append([(b_d[i][0] - b_min) / (b_max - b_min), (b_d[i][1] - d_min) / (d_max - d_min)])

        return torch.tensor(x)


class Agent:

    # n：问题维度  q_net：critic网络  q_target_net：critic网络的目标网络  model：actor网络
    # optimizer_critic：critic网络的优化器  optimizer_actor：actor网络的优化器  replay_memory：保存的样本集合
    # batch_size：采样大小  target_update：critic网络的目标网络更新频率  gamma：奖励衰减  loss：损失函数
    # map：根据问题维度n递归生成的离散动作选择集合
    def __init__(self, n, lr_critic, lr_actor, batch_size, target_update, gamma, path):
        if not path:
            self.n = n
            self.q_net = QNet(self.n)
            self.q_target_net = QNet(self.n)
            self.model = AgentNet()
            self.optimizer_critic = torch.optim.SGD(self.q_net.parameters(), lr=lr_critic)
            self.optimizer_actor = torch.optim.Adam(self.model.parameters(), lr=lr_actor)
            self.replay_memory = ReplayMemory()
            self.batch_size = batch_size
            self.count = 1
            self.target_update = target_update
            self.gamma = gamma
            self.loss = nn.MSELoss()
            self.map = self.generate_map(self.n)
            self.best_model = None
        else:
            self.model = torch.load(path)

    # state是元组，返回的action是列表，prob是tensor
    def interacte(self, state):
        prob = self.model(state)

        action = []
        for i in range(prob.shape[0]):
            if random.random() < prob[i].item():
                action.append(1)
            else:
                action.append(0)

        return action, prob

    # 返回的policy是元组
    def action_to_policy(self, state, action):

        policy = Resource().allocate(state, action)

        return policy

    # critic网络学习，传入的state和next_state是元组，action是列表，reward是值
    def critic_learn(self, state, action, reward, next_state):
        # 找到action在map中的位置
        for i in range(pow(2, self.n)):
            if action == self.map[i]:
                action = i
                break

        # 保存四元组
        self.replay_memory.push(state, action, reward, next_state)

        # 如果memory中的四元组个数大于batch_size，则可以采样并依据最优Bellman方程关系更新critic网络
        if len(self.replay_memory.memory) > self.batch_size:
            batch_state, batch_action, batch_reward, batch_next_state = \
                self.replay_memory.sample(self.batch_size)

            q = self.q_net(batch_state).gather(1, batch_action)
            q_max = torch.unsqueeze(self.q_target_net(batch_next_state).max(1).values, 1)
            q_target = batch_reward + self.gamma * q_max

            loss = self.loss(q, q_target)
            self.optimizer_critic.zero_grad()
            loss.backward()
            self.optimizer_critic.step()

        # 按照一定频率对q_target_net进行更新
        if self.count % self.target_update == 0:
            self.q_target_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    # actor网络学习，state是元组，action是列表
    def actor_learn(self, state, action):

        prob = self.model(state)
        prob_list = prob.tolist()

        # out为概率集合
        out = []
        for action_map in self.map:
            p = 1
            for i in range(self.n):
                p *= action_map[i] * prob_list[i][0] + (1 - action_map[i]) * (1 - prob_list[i][0])
            out.append(p)

        self.optimizer_actor.zero_grad()

        loss = torch.tensor(1, dtype=torch.float32).unsqueeze(0)
        for i in range(self.n):
            loss *= action[i] * prob[i] + (1 - action[i]) * (1 - prob[i])

        q_list = self.q_net(self.state_format_change(state)).tolist()
        v = 0

        for i in range(pow(2, self.n)):
            v += out[i] * q_list[i]

        for i in range(pow(2, self.n)):
            if action == self.map[i]:
                action = i
                break

        q = q_list[action]

        # a2c算法的更新公式，注意q - v
        loss = -1 * loss.log() * (q - v)

        loss.backward()

        self.optimizer_actor.step()

    def free_replay_memory(self):
        self.replay_memory.free()

    # 保存模型
    def save_model(self, path):
        torch.save(self.best_model, path)

    # 元组转可以输入critic网络的tensor形式
    def state_format_change(self, state):
        state = [state[0][i][0] for i in range(len(state[0]))] + [state[0][i][1] for i
                                                                  in range(len(state[0]))] + state[1]
        return torch.tensor(state)

    # 根据问题维度n递归产生map
    def generate_map(self, n):
        if n == 1:
            return [[0], [1]]
        map = []
        for list in self.generate_map(n - 1):
            map.append(list + [0])
            map.append(list + [1])
        return map


class Resource:
    def __init__(self):
        pass

    # 最优资源分配
    def allocate(self, state, action):

        policy = self.lagrange_multiplier(state[0], action)

        return policy

    # RRT条件下的拉格朗日乘子法
    def lagrange_multiplier(self, b_d, action):
        b_d_tmp = [[0, 0] for i in range(len(b_d))]
        w = 100.0
        s = 200.0
        for i in range(len(b_d)):
            b_d_tmp[i][0] = math.sqrt(b_d[i][0])
            b_d_tmp[i][1] = math.sqrt(b_d[i][1])
        b = [b_d_tmp[i][0] for i in range(len(b_d))]
        d = [b_d_tmp[i][1] for i in range(len(b_d))]
        W = [w * num / sum(b) for num in b]
        temp = [d[i] * action[i] for i in range(len(b_d))]
        if temp == [0 for i in range(len(b_d))]:
            S = temp
        else:
            S = [s * num / sum(temp) for num in temp]

        policy = (action, W, S)
        return policy


# 环境
class Env:
    def __init__(self):
        pass

    def max(self, a, b):
        if a > b:
            return a
        else:
            return b

    # 产生n个任务组成的集合
    def generate_state(self, n):
        b_d = []
        for i in range(n):
            b_d.append([0.086 * random.random() + 0.014, 1.36 * random.random() + 0.4])
        pt_sc = [self.max(0, random.gauss(0.04, 0.02)), 0.007]

        state = (b_d, pt_sc)
        return state

    # 状态转移
    def step(self, state, policy, n):
        b_d = state[0]
        pt_sc = state[1]
        # n个任务的数据量
        b = [b_d[i][0] for i in range(len(b_d))]
        # n个任务的cpu长度
        d = [b_d[i][1] for i in range(len(b_d))]
        # 云服务器的cpu速度
        pt = pt_sc[0]
        # 边缘服务器和云服务器间的延迟
        sc = pt_sc[1]

        # n个任务是否在边缘执行，1代表是，0代表否
        action = policy[0]
        # n个任务的带宽分配
        W = policy[1]
        # n个任务的边缘cpu速度分配
        S = policy[2]

        # 计算延迟
        delay = 0
        for i in range(len(action)):
            if S[i] == 0:
                S[i] = 1
            delay += b[i] / W[i] + action[i] * d[i] / S[i] + (1 - action[i]) * (d[i] / (sc * 10000) + pt)

        reward = -1 * delay

        next_state = self.generate_state(n)

        return reward, next_state


class Test():
    def __init__(self):
        pass

    # 对传入的agent进行测试，返回num次延迟的平均值
    def test_delay(self, agent, env, num, n):
        delay_sum = 0
        for i in range(num):
            state = env.generate_state(n)
            action, _ = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            reward, next_state = env.step(state, policy, n)
            delay = reward * -1
            delay_sum += delay

        return delay_sum / (num * n)


def main():
    n = 8
    agent = Agent(n=n, lr_critic=0.0000005, lr_actor=0.00005, batch_size=40, target_update=10, gamma=0.98, path=None)
    env = Env()
    step = 100
    epoch = 20
    min_avg_delay = 9999
    delay_list = []

    for i in range(epoch):
        state = env.generate_state(n)
        for j in range(step):
            action, prob = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            reward, next_state = env.step(state, policy, n)
            agent.critic_learn(state, action, reward, next_state)  # reward + 9
            agent.actor_learn(state, action)
            state = next_state

        print("==============epoch:" + str(i) + "==============")
        avg_delay = Test().test_delay(agent, env, 4000, n)
        if avg_delay < min_avg_delay:
            min_avg_delay = avg_delay
            agent.best_model = copy.deepcopy(agent.model)
        delay_list.append(avg_delay)
        print("客户端平均延迟(s):" + str(avg_delay))
        print("动作:" + str(action))
        print("动作概率:" + str(prob))
        agent.free_replay_memory()

    plt.plot(delay_list, c="red")
    plt.show()

    agent.save_model(
        r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")


if __name__ == "__main__":
    main()
