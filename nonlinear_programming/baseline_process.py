import copy
import random
from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *


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
                    action.append(7.0 / n)
                num = sum(action[0:n])
                for j in range(n):
                    if action[j] == 1:
                        action.append(10.0 / num)
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
                    action.append(random.random() * 4 + 1)
                for j in range(n):
                    if action[j] == 1:
                        action.append(random.random() * 4 + 1)
                    else:
                        action.append(0)
                sum_w = sum(action[n:2 * n])
                sum_b = sum(action[2 * n:3 * n])
                for j in range(n):
                    action[n + j] = action[n + j] / sum_w * 7
                    if sum_b != 0:
                        action[2 * n + j] = action[2 * n + j] / sum_b * 10

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
                          path=r"F:\PythonWorkspace\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
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

    def writeResult(self,n):
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
            content = file.readlines()

        with open(r"F:\MatlabWorkspace\nonlinear_programming\随机lm策略决策结果.txt", 'w+') as file:
            agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                          path=r"F:\PythonWorkspace\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
            env = Env()
            map = agent.generate_map(n)
            for i in range(4000):
                state = self.readState(content[i], n)
                action =random.sample(map,1)[0]
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
    RandomEquipartitionStrategy().writeResult(8)
    # RandomStrategy().writeResult(5)
    # GenerateState().writeResult(8)
    # BestStrategy().writeResult(8)
    # RandomLagrangeMultiplierStrategy().writeResult(3)


if __name__ == "__main__":
    main()
