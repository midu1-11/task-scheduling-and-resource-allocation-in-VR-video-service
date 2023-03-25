# 参考 https://blog.csdn.net/Er_Studying_Bai/article/details/128462002
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

map = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]


# DQN的第一个trick，记录所有的四元组
class ReplayMemory():
    def __init__(self):
        self.memory = []

    # 存储单个四元组
    def push(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    # 随机采样一个batch
    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = [], [], [], []
        for data in batch_data:
            batch_state.append(data[0].tolist())
            batch_action.append(data[1])
            batch_reward.append(data[2])
            batch_next_state.append(data[3].tolist())
        return torch.tensor(batch_state), torch.unsqueeze(torch.tensor(batch_action), 1), torch.unsqueeze(
            torch.tensor(batch_reward), 1), torch.tensor(
            batch_next_state)

    # 释放记录
    def free(self):
        self.memory = []


# Q网络
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 8)

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
        state = torch.tensor(state[0])
        pt_sc = torch.tensor(state[1])
        embed_x = torch.empty(0, 32)
        u = torch.empty(0, 1)
        embed_x_sum = 0
        for i in range(1, state.shape[0]):
            embed_x = torch.cat([embed_x, self.embed(state[i]).unsqueeze(0)], 0)
            embed_x_sum += self.embed(state[i])
        h = torch.cat([embed_x_sum / (state.shape[0] - 1), pt_sc], 0)

        wwww = self.V(nn.functional.tanh(self.W(torch.cat([embed_x[0], h], 0)))).unsqueeze(0)
        for i in range(1, state.shape[0]):
            u = torch.cat([u, self.V(nn.functional.tanh(self.W(torch.cat([embed_x[i-1], h], 0)))).unsqueeze(0)], 0)

        out = nn.functional.sigmoid(u)

        return out

    def state_change(self,state):
        b_max = 0
        b_min = 100
        d_max = 0
        d_min = 100
        for i in range(state.shape[0]-1):
            if state[i+1][0].item()>b_max:
                b_max=state[i+1][0].item()
            if state[i+1][0].item()<b_min:
                b_min=state[i+1][0].item()
            if state[i + 1][1].item() > d_max:
                d_max = state[i + 1][1].item()
            if state[i + 1][1].item() < d_min:
                d_min = state[i + 1][1].item()
        for i in range(state.shape[0]-1):
            state[i+1][0] = (state[i+1][0].item()-b_min)/(b_max-b_min)
            state[i+1][1] = (state[i+1][1].item()-d_min)/(d_max-d_min)
        return state


class Agent:

    # target_update：目标网络更新频率
    def __init__(self, lr_critic, lr_actor, batch_size, target_update, gamma, path):
        if not path:
            self.q_net = QNet()
            self.q_target_net = QNet()
            self.model = AgentNet()
            self.optimizer_critic = torch.optim.SGD(self.q_net.parameters(), lr=lr_critic)
            self.optimizer_actor = torch.optim.Adam(self.model.parameters(), lr=lr_actor)
            self.replay_memory = ReplayMemory()
            self.batch_size = batch_size
            self.count = 1
            self.target_update = target_update
            self.gamma = gamma
            self.loss = nn.MSELoss()
        else:
            self.model = torch.load(path)

    def state_change(self,state):
        state_buf = state
        state = torch.empty(4,2)
        state[0][0] = state_buf[6]
        state[0][1] = state_buf[7]
        state[1][0]=state_buf[0]
        state[1][1] = state_buf[3]
        state[2][0] = state_buf[1]
        state[2][1] = state_buf[4]
        state[3][0] = state_buf[2]
        state[3][1] = state_buf[5]

        return state

    def interacte(self, state):
        # state = self.state_change(state)

        prob = self.model(state)

        # prob[0] = 0.9
        # prob[1] = 0.1
        # prob[2] = 0.1

        action = []
        for i in range(prob.shape[0]):
            if random.random() < prob[i].item():
                action.append(1)
            else:
                action.append(0)

        return action, prob

    def action_to_policy(self, state, action):

        policy = Resource().allocate(state, torch.tensor(action))

        return torch.tensor(policy)

    def critic_learn(self, state, action, reward, next_state):
        for i in range(8):
            if action==map[i]:
                action = i
                break
        self.replay_memory.push(state, action, reward, next_state)

        # 如果memory中的四元组个数大于batch，则可以采样
        if len(self.replay_memory.memory) > self.batch_size:
            batch_state, batch_action, batch_reward, batch_next_state = self.replay_memory.sample(self.batch_size)

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

    def actor_learn(self, state, action):

        prob = self.model(self.state_change(state))
        prob_list = prob.tolist()
        out = [0,0,0,0,0,0,0,0] #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        out[0] = (1-prob_list[0][0])*(1-prob_list[1][0])*(1-prob_list[2][0])
        out[1] = (1 - prob_list[0][0]) * (1 - prob_list[1][0]) * prob_list[2][0]
        out[2] = (1 - prob_list[0][0]) * prob_list[1][0] * (1 - prob_list[2][0])
        out[3] = prob_list[0][0] * (1 - prob_list[1][0]) * (1 - prob_list[2][0])
        out[4] = (1 - prob_list[0][0]) * prob_list[1][0] * prob_list[2][0]
        out[5] = prob_list[0][0] * (1 - prob_list[1][0]) * prob_list[2][0]
        out[6] = prob_list[0][0] * prob_list[1][0] * (1 - prob_list[2][0])
        out[7] = prob_list[0][0] * prob_list[1][0] * prob_list[2][0]

        self.optimizer_actor.zero_grad()

        loss = torch.tensor(1,dtype=torch.float32).unsqueeze(0)
        x = action[0]*prob[0]+(1-action[0])*(1-prob[0])
        for i in range(3):
            loss *= action[i]*prob[i]+(1-action[i])*(1-prob[i])

        q_list = self.q_net(state).tolist()
        v = 0
        for i in range(8):
            v += out[i] * q_list[i]

        for i in range(8):
            if action==map[i]:
                action = i
                break

        q = q_list[action]
        loss = -1 * loss.log() * (q - v)

        loss.backward()

        self.optimizer_actor.step()

    def free_replay_memory(self):
        self.replay_memory.free()

    # 保存模型
    def save_model(self, path):
        torch.save(self.model, path)


class Resource:
    def __init__(self):
        pass

    def allocate(self, state, action):
        args1 = [state[0].item(), state[1].item(), state[2].item(), state[3].item(), state[4].item(), state[5].item(),
                 state[6].item(), state[7].item(), action[0].item(), action[1].item(), action[2].item()]
        policy = action.tolist()
        policy.extend(self.lagrange_multiplier(args1))

        return policy

    def lagrange_multiplier(self, args1):
        w = 7
        s = 10
        for i in range(6):
            args1[i] = math.sqrt(args1[i])
        b = [w * num / sum(args1[0:3]) for num in args1[0:3]]
        temp = [args1[i + 3] * args1[i + 8] for i in range(3)]
        if temp != [0, 0, 0]:
            d = [s * num / sum(temp) for num in temp]
        else:
            d = [0, 0, 0]
        policy = b
        policy.extend(d)
        return policy


class Env:
    def __init__(self):
        pass

    def generate_state(self,n):
        # state = [[abs(random.gauss(0.5, 1)), 5], [3 * random.random() + 0.5, 10 * random.random() + 4],
        #          [3 * random.random() + 0.5, 10 * random.random() + 4],
        #          [3 * random.random() + 0.5, 10 * random.random() + 4]]
        #
        # state = [3 * random.random() + 0.5, 3 * random.random() + 0.5, 3 * random.random() + 0.5,
        #          10 * random.random() + 4, 10 * random.random() + 4, 10 * random.random() + 4,
        #          abs(random.gauss(0.5, 1)), 5]

        b_d = []
        for i in range(n):
            b_d.append([3 * random.random() + 0.5, 10 * random.random() + 4])
        pt_sc = [abs(random.gauss(0.5, 1)), 5]

        state = (b_d,pt_sc)
        return state

    def step(self, state, policy):
        # 任务123的数据量
        b = [state[0].item(), state[1].item(), state[2].item()]
        # 任务123的cpu长度
        d = [state[3].item(), state[4].item(), state[5].item()]
        # 云服务器的cpu速度
        pt = state[6].item()
        # 边缘服务器和云服务器间的延迟
        sc = state[7].item()

        # 任务123是否在边缘执行，1代表是，0代表否
        x = [policy[0].item(), policy[1].item(), policy[2].item()]
        # 任务123的带宽分配
        w = [policy[3].item(), policy[4].item(), policy[5].item()]
        # 任务123的边缘cpu速度分配
        s = [policy[6].item(), policy[7].item(), policy[8].item()]

        # 计算延迟
        delay = 0
        for i in range(3):
            if s[i] == 0:
                s[i] = 1
            delay += b[i] / w[i] + x[i] * d[i] / s[i] + (1 - x[i]) * (d[i] / sc + pt)

        reward = -1 * delay

        next_state = self.generate_state()

        return reward, next_state


class Test():
    def __init__(self):
        pass

    # 对传入的agent进行测试，返回num次延迟的平均值
    def test_delay(self, agent, env, num):
        delay_sum = 0
        for i in range(num):
            state = env.generate_state()
            action, _ = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            reward, next_state = env.step(state, policy)
            delay = reward * -1
            delay_sum += delay

        return delay_sum / num


def main():
    agent = Agent(lr_critic=0.00001, lr_actor=0.0005, batch_size=40, target_update=10, gamma=0.98, path=None)
    env = Env()
    step = 100
    epoch = 20
    n = 3
    delay_list = []

    for i in range(epoch):
        state = env.generate_state()
        for j in range(step):
            action, prob = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            reward, next_state = env.step(state, policy)
            agent.critic_learn(state, action, reward, next_state)  # reward + 9
            agent.actor_learn(state, action)
            state = next_state

        print("==============epoch:" + str(i) + "==============")
        avg_delay = Test().test_delay(agent, env, 1000)
        delay_list.append(avg_delay)
        print("平均延迟:" + str(avg_delay))
        print("动作:" + str(action))
        print("动作概率:" + str(prob))
        agent.free_replay_memory()

    plt.plot(delay_list, c="red")
    plt.show()

    # agent.save_model(r"F:\PythonWorkspace\nonlinear_programming\ac_lm\ACLMModel.pkl")


if __name__ == "__main__":
    main()
