import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import math

import os
from scipy.optimize import minimize
import time

map = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]


class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
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
        out = nn.functional.softmax(out)

        return out


class Agent:
    def __init__(self, lr, path):
        if not path:
            self.model = AgentNet()
        else:
            self.model = torch.load(path)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def interacte(self, state):
        out = self.model(state)
        prob = out.clone()
        for i in range(1, 8):
            out[i] += out[i - 1]
        p = random.random()
        for i in range(8):
            if i == 0:
                if p >= 0 and p < out[i].item():
                    action = map[i]
                    break
            else:
                if p >= out[i - 1].item() and p < out[i].item():
                    action = map[i]
                    break
        return torch.tensor(action), prob

    def interacte_with_random(self, state):
        action = map[int(random.random() * 8)]
        if random.random() > 0.2:
            out = self.model(state)
            for i in range(1, 8):
                out[i] += out[i - 1]
            p = random.random()
            for i in range(8):
                if i == 0:
                    if p >= 0 and p < out[i].item():
                        action = map[i]
                        break
                else:
                    if p >= out[i - 1].item() and p < out[i].item():
                        action = map[i]
                        break
        return torch.tensor(action)

    def action_to_policy(self, state, action):

        policy = Resource().allocate(state, action)

        return torch.tensor(policy)

    def learn(self, state, action, reward):
        out = self.model(state)
        self.optimizer.zero_grad()

        for i in range(8):
            if map[i] == action.tolist():
                loss = out[i]

        loss = -1 * loss.log() * reward

        loss.backward()

        self.optimizer.step()

    # 保存模型
    def save_model(self, path):
        torch.save(self.model, path)


class Env:
    def __init__(self):
        pass

    def generate_state(self):
        state = [3 * random.random() + 0.5, 3 * random.random() + 0.5, 3 * random.random() + 0.5,
                 10 * random.random() + 4, 10 * random.random() + 4, 10 * random.random() + 4,
                 abs(random.gauss(0.5, 1)), 5]

        return torch.tensor(state)

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

        return reward

    def state_value(self, agent, state):
        action = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        reward_sum = 0
        for i in range(8):
            policy = agent.action_to_policy(state, action[i])
            reward_sum += self.step(state, policy)
        return reward_sum / 8


class Resource:
    def __init__(self):
        pass

    def allocate(self, state, action):
        # args1 = (state[0].item(), state[1].item(), state[2].item(), state[3].item(), state[4].item(), state[5].item(),
        #          state[6].item(), state[7].item(), action[0].item(), action[1].item(), action[2].item())
        # args2 = (7, 10)
        # cons = self.con(args2)
        # x0 = np.array((2.3, 2.3, 2.3, 3.3, 3.3, 3.3))  # 初值
        # # res = minimize(fun(args1), x0, method='SLSQP', constraints=cons)
        # start = time.time()
        # res = minimize(self.fun(args1), x0, constraints=cons)
        # end = time.time()
        # # print("time=" + str(end - start))
        #
        # policy = action.tolist()
        # policy.extend(np.array(res.x))

        args1 = [state[0].item(), state[1].item(), state[2].item(), state[3].item(), state[4].item(), state[5].item(),
                 state[6].item(), state[7].item(), action[0].item(), action[1].item(), action[2].item()]
        policy = action.tolist()
        policy.extend(self.lagrange_multiplier(args1))

        return policy

    def con(self, args2):
        w, s = args2
        cons = ({'type': 'eq', 'fun': lambda x: -x[0] - x[1] - x[2] + w},
                {'type': 'eq', 'fun': lambda x: -x[3] - x[4] - x[5] + s},
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.000001},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.000001},
                {'type': 'ineq', 'fun': lambda x: x[2] - 0.000001},
                {'type': 'ineq', 'fun': lambda x: x[3] - 0.000001},
                {'type': 'ineq', 'fun': lambda x: x[4] - 0.000001},
                {'type': 'ineq', 'fun': lambda x: x[5] - 0.000001})
        return cons

    # 目标函数
    def fun(self, args1):
        b1, b2, b3, d1, d2, d3, pt, sc, x1, x2, x3 = args1
        r = lambda x: (
                b1 / x[0] + b2 / x[1] + b3 / x[2] + x1 * d1 / x[3] + x2 * d2 / x[4] + x3 * d3 / x[5] + (1 - x1) * (
                d1 / sc + pt) + (1 - x2) * (d2 / sc + pt) + (1 - x3) * (d3 / sc + pt))
        return r

    def lagrange_multiplier(self, args1):
        w = 7
        s = 10
        for i in range(6):
            args1[i] = math.sqrt(args1[i])
        b = [w * num / sum(args1[0:3]) for num in args1[0:3]]
        temp = [args1[i + 3] * args1[i + 8] for i in range(3)]
        if temp!=[0,0,0]:
            d = [s * num / sum(temp) for num in temp]
        else:
            d = [0,0,0]
        policy = b
        policy.extend(d)
        return policy


def test_agent_delay(agent, path):
    with open(path, encoding='utf-8') as file:
        file_content = file.readlines()
    delay_sum = 0
    env = Env()
    for i in range(500):
        content = [float(numeric_string) for numeric_string in file_content[i].split(' ')]
        state = torch.tensor(content[0:8])
        action, prob = agent.interacte(state)
        policy = agent.action_to_policy(state, action)
        delay_sum += -1 * env.step(state, policy)

    return delay_sum / 500.0


def test_agent_choice(agent, path):
    with open(path, encoding='utf-8') as file:
        file_content = file.readlines()
    sum = 0
    for i in range(500):
        content = [float(numeric_string) for numeric_string in file_content[i].split(' ')]
        state = torch.tensor(content[0:8])
        action = agent.interacte(state)
        optimal_action = content[8:11]
        if action[0].item() == optimal_action[0] and action[1].item() == optimal_action[1] and action[2].item() == \
                optimal_action[2]:
            sum += 1
        # print("最优=(" + str(optimal_action[0]) + "," + str(optimal_action[1]) + "," + str(optimal_action[2]) + ")")
        # print("智能体=(" + str(action[0].item()) + "," + str(action[1].item()) + "," + str(action[2].item()) + ")")
    return sum / 500.0


def main():
    agent = Agent(0.0001,None)  # 0.0001
    env = Env()
    step = 100
    epoch = 30

    for i in range(step * epoch):
        state = env.generate_state()
        action, prob = agent.interacte(state)
        policy = agent.action_to_policy(state, action)
        reward = env.step(state, policy)
        state_value = env.state_value(agent, state)
        # agent.learn(state, action, (reward - state_value)/abs(state_value))
        agent.learn(state, action, (reward - state_value))
        # agent.learn(state, action, reward)
        # print(action)

        if i % step == 0 and i != 0:
            # accuracy = test_agent(agent, "F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt")
            avg_delay = test_agent_delay(agent, "F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt")
            print("=====================================")
            # print("accuracy=" + str(accuracy))
            print(action)
            print(prob)
            print(reward - state_value)
            print("平均延迟：" + str(avg_delay))
            print("=====================================")

    agent.save_model(r"F:\PythonWorkspace\nonlinear_programming\tss\TSSModel.pkl")


if __name__ == "__main__":
    main()
