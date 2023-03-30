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
    if n <= 12:
        with open("F:\\MatlabWorkspace\\nonlinear_programming\\最优策略决策结果.txt", encoding='utf-8') as best_file:
            best_content = best_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机均分策略决策结果.txt",
              encoding='utf-8') as randomEquipartitionStrategy_file:
        randomEquipartitionStrategy_content = randomEquipartitionStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机策略决策结果.txt", encoding='utf-8') as randomStrategy_file:
        randomStrategy_content = randomStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机lm策略决策结果.txt",
              encoding='utf-8') as randomLagrangeMultiplierStrategy_file:
        randomLagrangeMultiplierStrategy_content = randomLagrangeMultiplierStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\GAL策略决策结果.txt", encoding='utf-8') as GALStrategy_file:
        GALStrategy_content = GALStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\云lm策略决策结果.txt",
              encoding='utf-8') as CloudLagrangeMultiplierStrategy_file:
        CloudLagrangeMultiplierStrategy_content = CloudLagrangeMultiplierStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\边缘lm策略决策结果.txt",
              encoding='utf-8') as EdgeLagrangeMultiplierStrategy_file:
        EdgeLagrangeMultiplierStrategy_content = EdgeLagrangeMultiplierStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\阈值比例策略决策结果.txt",
              encoding='utf-8') as ThresholdProportionalStrategy_file:
        ThresholdProportionalStrategy_content = ThresholdProportionalStrategy_file.readlines()

    if n <= 12:
        best_delay_list = []
    randomEquipartitionStrategy_delay_list = []
    randomStrategy_delay_list = []
    randomLagrangeMultiplierStrategy_delay_list = []
    GALStrategy_delay_list = []
    CloudLagrangeMultiplierStrategy_delay_list = []
    EdgeLagrangeMultiplierStrategy_delay_list = []
    ThresholdProportionalStrategy_delay_list = []

    for i in range(0, 4000):
        state = Compare().readState(content[i], n)

        if n <= 12:
            best_policy = Compare().readPolicy(best_content[i], n)
        randomEquipartitionStrategy_policy = Compare().readPolicy(randomEquipartitionStrategy_content[i], n)
        randomStrategy_policy = Compare().readPolicy(randomStrategy_content[i], n)
        randomLagrangeMultiplierStrategy_policy = Compare().readPolicy(randomLagrangeMultiplierStrategy_content[i], n)
        GALStrategy_policy = Compare().readPolicy(GALStrategy_content[i], n)
        CloudLagrangeMultiplierStrategy_policy = Compare().readPolicy(CloudLagrangeMultiplierStrategy_content[i], n)
        EdgeLagrangeMultiplierStrategy_policy = Compare().readPolicy(EdgeLagrangeMultiplierStrategy_content[i], n)
        ThresholdProportionalStrategy_policy = Compare().readPolicy(ThresholdProportionalStrategy_content[i], n)

        if n <= 12:
            best_delay, _ = env.step(state, best_policy, n)
            best_delay *= -1
        randomEquipartitionStrategy_delay, _ = env.step(state, randomEquipartitionStrategy_policy, n)
        randomEquipartitionStrategy_delay *= -1
        randomStrategy_delay, _ = env.step(state, randomStrategy_policy, n)
        randomStrategy_delay *= -1
        randomLagrangeMultiplierStrategy_delay, _ = env.step(state, randomLagrangeMultiplierStrategy_policy, n)
        randomLagrangeMultiplierStrategy_delay *= -1
        GALStrategy_delay, _ = env.step(state, GALStrategy_policy, n)
        GALStrategy_delay *= -1
        CloudLagrangeMultiplierStrategy_delay, _ = env.step(state, CloudLagrangeMultiplierStrategy_policy, n)
        CloudLagrangeMultiplierStrategy_delay *= -1
        EdgeLagrangeMultiplierStrategy_delay, _ = env.step(state, EdgeLagrangeMultiplierStrategy_policy, n)
        EdgeLagrangeMultiplierStrategy_delay *= -1
        ThresholdProportionalStrategy_delay, _ = env.step(state, ThresholdProportionalStrategy_policy, n)
        ThresholdProportionalStrategy_delay *= -1

        if n <= 12:
            best_delay_list.append(best_delay)
        randomEquipartitionStrategy_delay_list.append(randomEquipartitionStrategy_delay)
        randomStrategy_delay_list.append(randomStrategy_delay)
        randomLagrangeMultiplierStrategy_delay_list.append(randomLagrangeMultiplierStrategy_delay)
        GALStrategy_delay_list.append(GALStrategy_delay)
        CloudLagrangeMultiplierStrategy_delay_list.append(CloudLagrangeMultiplierStrategy_delay)
        EdgeLagrangeMultiplierStrategy_delay_list.append(EdgeLagrangeMultiplierStrategy_delay)
        ThresholdProportionalStrategy_delay_list.append(ThresholdProportionalStrategy_delay)

    # 画图
    plt.figure()
    # plt.plot(best_delay_list, c="red")
    plt.plot(randomEquipartitionStrategy_delay_list, c="yellow")
    plt.plot(randomStrategy_delay_list, c="yellow")
    plt.plot(randomLagrangeMultiplierStrategy_delay_list, c="black")
    plt.plot(GALStrategy_delay_list, c="black")
    plt.legend(["随机均分策略", "随机策略", "随机lm策略", "GAL策略"])
    plt.xlabel("时隙")
    plt.ylabel("当前时隙任务平均延迟")
    plt.show()

    # 打印客户端平均延迟对比
    print("=============客户端平均延迟(ms)=============")
    if n <= 12:
        print("枚举最优策略:" + str(sum(best_delay_list) / (4000 * n) * 1000))
    print("GAL策略:" + str(sum(GALStrategy_delay_list) / (4000 * n) * 1000))
    print("随机lm策略:" + str(sum(randomLagrangeMultiplierStrategy_delay_list) / (4000 * n) * 1000))
    # print("云lm策略:" + str(sum(CloudLagrangeMultiplierStrategy_delay_list) / (4000 * n) * 1000))
    # print("边缘lm策略:" + str(sum(EdgeLagrangeMultiplierStrategy_delay_list) / (4000 * n) * 1000))
    print("阈值比例策略:" + str(sum(ThresholdProportionalStrategy_delay_list) / (4000 * n) * 1000))
    print("随机均分策略:" + str(sum(randomEquipartitionStrategy_delay_list) / (4000 * n) * 1000))
    print("完全随机策略:" + str(sum(randomStrategy_delay_list) / (4000 * n) * 1000))


if __name__ == "__main__":
    main(3)
