from gnn_a2c_lm.graph_neural_network_advantage_actor_critic_lagrange_multiplier import *


def readState(line, n):
    line = line.split(' ')
    b_d = []
    for i in range(n):
        b_d.append([(float)(line[2 * i]), (float)(line[2 * i + 1])])
    pt_sc = [(float)(line[2 * n]), (float)(line[2 * n + 1])]
    state = (b_d, pt_sc)
    return state

def main(n):
    agent = Agent(n=n, lr_critic=0, lr_actor=0, batch_size=0, target_update=0, gamma=0,
                  path=r"F:\PythonWorkspace\nonlinear_programming\gnn_a2c_lm\GALModel.pkl")
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试样本.txt", encoding='utf-8') as file:
        content = file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\GAL策略决策结果.txt", 'w+') as GALfile:
        for i in range(4000):
            state = readState(content[i], n)
            action, _ = agent.interacte(state)
            policy = agent.action_to_policy(state, action)
            saveContent = ""
            savePolicy = policy[0] + policy[1] + policy[2]
            for i in range(len(savePolicy)):
                saveContent += str(savePolicy[i])
                if i != len(savePolicy) - 1:
                    saveContent += " "
                else:
                    saveContent += "\n"
            GALfile.write(saveContent)

if __name__ == "__main__":
    main(3)