from two_step_solution import *

map = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]


def main():
    # 读入测试集信息
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt", encoding='utf-8') as file:
        content = file.readlines()

    with open(r"F:\MatlabWorkspace\nonlinear_programming\TSS决策结果.txt", 'w+') as file:
        agent = Agent(0, "./TSSModel.pkl")

        # 4000条测试集数据
        for i in range(4000):
            state = []
            for j in range(8):
                state.append((float)(content[i].split(' ')[j]))
            action, prob = agent.interacte(torch.tensor(state))
            policy = agent.action_to_policy(torch.tensor(state), action)

            file.write(str(policy[0].item()) + " " + str(policy[1].item()) + " " + str(policy[2].item()) + " " + str(
                policy[3].item()) + " " + str(policy[4].item()) + " " + str(policy[5].item()) + " " + str(
                policy[6].item()) + " " + str(policy[7].item()) + " " + str(policy[8].item()) + "\n")


if __name__ == "__main__":
    main()
