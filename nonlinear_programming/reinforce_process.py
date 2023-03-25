import torch
import random
import numpy as np
import math
import torch.nn as nn


class MlpNet(nn.Module):

    def __init__(self):
        super(MlpNet, self).__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 17)

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


def main():
    # 加载模型
    new_model = torch.load('./PGModel.pkl')

    # 读入测试集信息
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt", encoding='utf-8') as file:
        content = file.readlines()

    with open(r"F:\MatlabWorkspace\nonlinear_programming\PG决策结果.txt", 'w+') as file:
        # 4000条测试集数据
        for i in range(4000):
            state = []
            action = []
            for j in range(8):
                state.append((float)(content[i].split(' ')[j]))

            out1, out2, out3 = new_model(torch.tensor(state))

            for i in range(3):
                if out1[i] > random.random():
                    action.append(1)
                else:
                    action.append(0)

            for i in range(6):
                action.append(np.random.normal(out2[2 * i].item(), math.sqrt(math.e ** out2[2 * i + 1].item())))

            action.append(np.random.normal(out3[0].item(), math.sqrt(math.e ** out3[1].item())))

            u = 8.31 * nn.functional.sigmoid(torch.tensor(action[9])).item() # 7.16
            u = math.e ** (u / 4) - 1

            if u < 1:
                u = 1

            sum_w = 0
            sum_s = 0
            for i in range(3):
                action[3 + i] = math.e ** (action[3 + i] / u)
                sum_w += action[3 + i]
                action[6 + i] = math.e ** (action[6 + i] / u)
                if action[i] == 1:
                    sum_s += action[6 + i]

            if sum_s == 0:
                sum_s = 1

            for i in range(3):
                action[3 + i] = action[3 + i] / sum_w * 7
                action[6 + i] = action[6 + i] / sum_s * 10

            if action[0] == 0:
                action[6] = 0
            if action[1] == 0:
                action[7] = 0
            if action[2] == 0:
                action[8] = 0

            file.write(str(action[0]) + " " + str(action[1]) + " " + str(action[2]) + " " + str(action[3]) + " " + str(
                action[4]) + " " + str(action[5]) + " " + str(action[6]) + " " + str(action[7]) + " " + str(
                action[8]) + "\n")


if __name__ == "__main__":
    main()
