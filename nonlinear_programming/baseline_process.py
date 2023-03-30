import copy
import random
from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *

w = 0
s = 0


class RandomEquipartitionStrategy:
    def __init__(self):
        pass

    def writeResult(self, n):
        with open(r"F:\MatlabWorkspace\nonlinear_programming\随机均分策略决策结果.txt", 'w+') as file:
            for i in range(4000):
                action = []
                for j in range(n):
                    action.append(round(random.random()))
                for j in range(n):
                    action.append(w / n)
                num = sum(action[0:n])
                for j in range(n):
                    if action[j] == 1:
                        action.append(s / num)
                    else:
                        action.append(0)
                content = ""
                for i in range(3 * n):
                    content += str(action[i])
                    if i != 3 * n - 1:
                        content += " "
                    else:
                        content += "\n"
                file.write(content)


class ThresholdProportionalStrategy:
    def __init__(self):
        pass

    def readState(self, line, n):
        line = line.split(' ')
        b_d = []
        for i in range(n):
            b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
        pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
        state = (b_d, pt_sc)
        return state

    def writeResult(self, n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()
        with open(r"F:\MatlabWorkspace\nonlinear_programming\阈值比例策略决策结果.txt", 'w+') as file:
            for i in range(4000):
                state = self.readState(content[i], n)
                b_d = state[0]
                action = []
                for j in range(n):
                    if b_d[j][1] > 0.8:
                        action.append(0)
                    else:
                        action.append(1)
                sum_b = 0
                sum_d = 0
                for j in range(n):
                    sum_b += b_d[j][0]
                    if action[j] == 1:
                        sum_d += b_d[j][1]
                for j in range(n):
                    action.append(b_d[j][0] / sum_b * w)
                for j in range(n):
                    if action[j] == 1:
                        action.append(b_d[j][1] / sum_d * s)
                    else:
                        action.append(0)

                saveContent = ""
                for i in range(3 * n):
                    saveContent += str(action[i])
                    if i != 3 * n - 1:
                        saveContent += " "
                    else:
                        saveContent += "\n"
                file.write(saveContent)


class RandomStrategy:
    def __init__(self):
        pass

    def writeResult(self, n):
        with open(r"F:\MatlabWorkspace\nonlinear_programming\随机策略决策结果.txt", 'w+') as file:
            for i in range(4000):
                action = []
                for j in range(n):
                    action.append(round(random.random()))
                for j in range(n):
                    action.append(random.random() * 4 + 0.1)
                for j in range(n):
                    if action[j] == 1:
                        action.append(random.random() * 4 + 0.1)
                    else:
                        action.append(0)
                sum_w = sum(action[n:2 * n])
                sum_b = sum(action[2 * n:3 * n])
                for j in range(n):
                    action[n + j] = action[n + j] / sum_w * w
                    if sum_b != 0:
                        action[2 * n + j] = action[2 * n + j] / sum_b * s

                content = ""
                for i in range(3 * n):
                    content += str(action[i])
                    if i != 3 * n - 1:
                        content += " "
                    else:
                        content += "\n"
                file.write(content)


class GenerateState:
    def __init__(self):
        pass

    def writeResult(self, n):
        env = Env()
        with open(r"F:\MatlabWorkspace\nonlinear_programming\测试样本.txt", 'w+') as file:
            for i in range(4000):
                state = env.generate_state(n)
                content = ""
                for i in range(n):
                    content += str(state[0][i][0]) + " "
                    content += str(state[0][i][1]) + " "
                content += str(state[1][0])
                content += " "
                content += str(state[1][1])
                content += "\n"
                file.write(content)


class BestStrategy:
    def __init__(self):
        pass

    def readState(self, line, n):
        line = line.split(' ')
        b_d = []
        for i in range(n):
            b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
        pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
        state = (b_d, pt_sc)
        return state

    def writeResult(self, n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()

        with open(r"F:\MatlabWorkspace\nonlinear_programming\最优策略决策结果.txt", 'w+') as file:
            agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                          path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
            env = Env()
            map = agent.generate_map(n)
            for i in range(4000):
                state = self.readState(content[i], n)
                minDelay = 9999
                minPolicy = None
                for action in map:
                    policy = agent.action_to_policy(state, action)
                    delay, _ = env.step(state, copy.deepcopy(policy), n)
                    delay *= -1
                    if delay < minDelay:
                        minDelay = delay
                        minPolicy = policy
                saveContent = ""
                minPolicy = minPolicy[0] + minPolicy[1] + minPolicy[2]
                for i in range(len(minPolicy)):
                    saveContent += str(minPolicy[i])
                    if i != len(minPolicy) - 1:
                        saveContent += " "
                    else:
                        saveContent += "\n"
                file.write(saveContent)


class RandomLagrangeMultiplierStrategy:
    def __init__(self):
        pass

    def readState(self, line, n):
        line = line.split(' ')
        b_d = []
        for i in range(n):
            b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
        pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
        state = (b_d, pt_sc)
        return state

    def writeResult(self, n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()

        with open(r"F:\MatlabWorkspace\nonlinear_programming\随机lm策略决策结果.txt", 'w+') as file:
            agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                          path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
            env = Env()
            for i in range(4000):
                state = self.readState(content[i], n)
                action = []
                for j in range(n):
                    action.append(round(random.random()))
                policy = agent.action_to_policy(state, action)
                saveContent = ""
                savePolicy = policy[0] + policy[1] + policy[2]
                for i in range(len(savePolicy)):
                    saveContent += str(savePolicy[i])
                    if i != len(savePolicy) - 1:
                        saveContent += " "
                    else:
                        saveContent += "\n"
                file.write(saveContent)


class CloudLagrangeMultiplierStrategy:
    def __init__(self):
        pass

    def readState(self, line, n):
        line = line.split(' ')
        b_d = []
        for i in range(n):
            b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
        pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
        state = (b_d, pt_sc)
        return state

    def writeResult(self, n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()

        with open(r"F:\MatlabWorkspace\nonlinear_programming\云lm策略决策结果.txt", 'w+') as file:
            agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                          path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
            env = Env()
            for i in range(4000):
                state = self.readState(content[i], n)
                action = [0 for j in range(n)]
                policy = agent.action_to_policy(state, action)
                saveContent = ""
                savePolicy = policy[0] + policy[1] + policy[2]
                for i in range(len(savePolicy)):
                    saveContent += str(savePolicy[i])
                    if i != len(savePolicy) - 1:
                        saveContent += " "
                    else:
                        saveContent += "\n"
                file.write(saveContent)


class EdgeLagrangeMultiplierStrategy:
    def __init__(self):
        pass

    def readState(self, line, n):
        line = line.split(' ')
        b_d = []
        for i in range(n):
            b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
        pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
        state = (b_d, pt_sc)
        return state

    def writeResult(self, n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()

        with open(r"F:\MatlabWorkspace\nonlinear_programming\边缘lm策略决策结果.txt", 'w+') as file:
            agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                          path=r"F:\Bupt\task-scheduling-and-resource-allocation-in-VR-video-service\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
            env = Env()
            for i in range(4000):
                state = self.readState(content[i], n)
                action = [1 for j in range(n)]
                policy = agent.action_to_policy(state, action)
                saveContent = ""
                savePolicy = policy[0] + policy[1] + policy[2]
                for i in range(len(savePolicy)):
                    saveContent += str(savePolicy[i])
                    if i != len(savePolicy) - 1:
                        saveContent += " "
                    else:
                        saveContent += "\n"
                file.write(saveContent)


def main():
    n = 3
    global w, s
    w = 100.0
    s = 200.0
    GenerateState().writeResult(n)
    RandomEquipartitionStrategy().writeResult(n)
    RandomStrategy().writeResult(n)
    ThresholdProportionalStrategy().writeResult(n)
    if n <= 12:
        BestStrategy().writeResult(n)
    RandomLagrangeMultiplierStrategy().writeResult(n)
    CloudLagrangeMultiplierStrategy().writeResult(n)
    EdgeLagrangeMultiplierStrategy().writeResult(n)


if __name__ == "__main__":
    main()
