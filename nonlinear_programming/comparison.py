import matplotlib.pyplot as plt
from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


class Compare:
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

    def readPolicy(self, line, n):
        line = line.split(' ')
        action = []
        W = []
        S = []
        for i in range(n):
            action.append((float)(line[i]))
            W.append((float)(line[i + n]))
            S.append((float)(line[i + 2 * n]))
        policy = (action, W, S)
        return policy


def main(n):
    env = Env()

    with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
        content = file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\最优策略决策结果.txt", encoding='utf-8') as best_file:
        best_content = best_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机均分策略决策结果.txt", encoding='utf-8') as randomEquipartitionStrategy_file:
        randomEquipartitionStrategy_content = randomEquipartitionStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\GAL策略决策结果.txt", encoding='utf-8') as GALStrategy_file:
        GALStrategy_content = GALStrategy_file.readlines()

    best_delay_list = []
    randomEquipartitionStrategy_delay_list = []
    GALStrategy_delay_list = []

    for i in range(0,100):
        state = Compare().readState(content[i], n)

        best_policy = Compare().readPolicy(best_content[i], n)
        randomEquipartitionStrategy_policy = Compare().readPolicy(randomEquipartitionStrategy_content[i], n)
        GALStrategy_policy = Compare().readPolicy(GALStrategy_content[i], n)

        best_delay, _ = env.step(state, best_policy, n)
        best_delay *= -1
        randomEquipartitionStrategy_delay, _ = env.step(state, randomEquipartitionStrategy_policy, n)
        randomEquipartitionStrategy_delay *= -1
        GALStrategy_delay, _ = env.step(state, GALStrategy_policy, n)
        GALStrategy_delay *= -1

        best_delay_list.append(best_delay)
        randomEquipartitionStrategy_delay_list.append(randomEquipartitionStrategy_delay)
        GALStrategy_delay_list.append(GALStrategy_delay)


    # 画图
    plt.figure()
    plt.plot(best_delay_list, c="red")
    plt.plot(randomEquipartitionStrategy_delay_list, c="yellow")
    plt.plot(GALStrategy_delay_list, c="black")
    plt.legend(["最优策略","随机均分策略","GAL策略"])
    plt.xlabel("时隙")
    plt.ylabel("当前时隙任务平均延迟")
    plt.show()

    # 打印平均延迟对比
    print("最优策略:"+str(sum(best_delay_list)/4000))
    print("随机均分策略:" + str(sum(randomEquipartitionStrategy_delay_list) / 4000))
    print("GAL策略:" + str(sum(GALStrategy_delay_list) / 4000))

if __name__ == "__main__":
    main(8)
