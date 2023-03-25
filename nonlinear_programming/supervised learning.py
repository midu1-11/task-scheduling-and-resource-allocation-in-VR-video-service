import torch
import time
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 分类网络，输出3个任务最优解中在边缘执行概率的拟合值
class classify_net(torch.nn.Module):  # 继承torch.nn.Module构建自己的网络
    def __init__(self):
        super(classify_net, self).__init__()
        self.layer1 = torch.nn.Linear(8, 32)  # 隐藏层
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, 64)
        self.layer_out = torch.nn.Linear(64, 3)  # 输出层

    def forward(self, input):
        # 第一个隐藏层
        out = self.layer1(input)
        out = torch.relu(out)  # 隐藏层的激活函数设置成Relu函数
        # 第二个隐藏层
        out = self.layer2(out)
        out = torch.relu(out)
        # 第三个隐藏层
        out = self.layer3(out)
        out = torch.relu(out)
        # 输出层
        out = self.layer_out(out)  # 输出层直接输出
        out = torch.sigmoid(out)
        return out

# 回归网络，输出3个任务最优解中带宽分配和边缘cpu速度分配的拟合值
class regression_net(torch.nn.Module):  # 继承torch.nn.Module构建自己的网络
    def __init__(self):
        super(regression_net, self).__init__()
        self.layer1 = torch.nn.Linear(8, 32)  # 隐藏层
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, 64)
        self.layer_out = torch.nn.Linear(64, 6)  # 输出层

    def forward(self, input):
        # 第一个隐藏层
        out = self.layer1(input)
        out = torch.relu(out)  # 隐藏层的激活函数设置成Relu函数
        # 第二个隐藏层
        out = self.layer2(out)
        out = torch.relu(out)
        # 第三个隐藏层
        out = self.layer3(out)
        out = torch.relu(out)
        # 输出层
        out = self.layer_out(out)  # 输出层直接输出
        return out


# 训练分类网络
def train_classify(X, Y, net, X_test, Y_test):
    X, Y = X.to(device), Y.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)  # 设置梯度下降的算法为SGD算法，学习率为0.03。SGD使随机梯度下降
    loss_func = torch.nn.MSELoss()
    t0 = time.time_ns()
    loss = 0
    loss_list = []
    accuracy_list = []
    for i in range(10000):
        Y_hat = net(X)  # 前馈过程
        loss = loss_func(Y_hat, Y)  # 计算损失函数
        # 反向传播计算梯度并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔1000步打印一下loss的值
        if i % 100 == 0:
            loss_list.append(loss.data.item())
            print("step=", i, "loss=", loss.data.item())
            accuracy = test(X_test, Y_test, net)
            accuracy_list.append(accuracy)
            print("accuracy=" + accuracy)
    t1 = time.time_ns()
    return loss_list, accuracy_list


# 训练回归网络
def train_regression(X, Y, net):
    X, Y = X.to(device), Y.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)  # 设置梯度下降的算法为SGD算法，学习率为0.03。SGD使随机梯度下降
    loss_func = torch.nn.MSELoss()
    t0 = time.time_ns()
    loss = 0
    loss_list = []
    for i in range(10000):
        Y_hat = net(X)  # 前馈过程
        loss = loss_func(Y_hat, Y)  # 计算损失函数
        # 反向传播计算梯度并更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔1000步打印一下loss的值
        if i % 100 == 0:
            loss_list.append(loss.data.item())
            print("step=", i, "loss=", loss.data.item())
    t1 = time.time_ns()
    return loss_list


# 加载txt文件数据
def load_data(path):
    with open(path, encoding='utf-8') as file:
        content = file.readlines()
    X = np.ones(shape=(len(content), 8), dtype="float64")
    Y = np.ones(shape=(len(content), 9), dtype="float64")
    ###逐行读取数据
    for j in range(len(content)):
        for i in range(8):
            X[j][i] = content[j].split(' ')[i]
        for i in range(8, 17):
            Y[j][i - 8] = content[j].split(' ')[i]

    return torch.from_numpy(X).type(torch.FloatTensor), torch.from_numpy(Y).type(torch.FloatTensor)


# 加载测试集数据
def load_test_data(path):
    return load_data(path)

# 加载训练集数据
def load_train_data(path):
    return load_data(path)


# 返回分类网络在测试集的准确率
def test(X_test, Y_test, classify_net):
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    Y_classify_test = classify_net(X_test)
    # Y_regression_test = regression_net(X_test)

    dim0, dim1 = Y_classify_test.shape
    count = 0

    for i in range(dim0):
        if round(Y_classify_test[i][0].item()) == Y_test[i][0] and round(Y_classify_test[i][1].item()) == Y_test[i][
            1] and round(Y_classify_test[i][2].item()) == Y_test[i][2]:
            count += 1

    return str(count / dim0)


# 保存测试集的决策结果，包括3个任务是否在边缘执行，带宽分配，边缘cpu速度分配9个决策量
def save_result(classify_net, regression_net, X_test, Y_test, path):
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    Y_classify_test = classify_net(X_test)
    Y_regression_test = regression_net(X_test)
    dim0, dim1 = Y_classify_test.shape
    s = ""

    with open(path, 'w+') as file:
        for i in range(dim0):
            sum1 = 0
            sum2 = 0
            for j in range(3):
                Y_classify_test[i][j] = round(Y_classify_test[i][j].item())
                if Y_classify_test[i][j] == 0:
                    Y_regression_test[i][3 + j] = 0
                sum1 += Y_regression_test[i][3 + j]
                sum2 += Y_regression_test[i][j]

            if sum1 != 0:
                for j in range(3):
                    Y_regression_test[i][3 + j] = Y_regression_test[i][3 + j].item() / sum1 * 10

            for j in range(3):
                Y_regression_test[i][j] = Y_regression_test[i][j].item() / sum2 * 7

            file.write(str(Y_classify_test[i][0].item()) + " " + str(Y_classify_test[i][1].item()) + " " + str(
                Y_classify_test[i][2].item()) + " " + str(Y_regression_test[i][0].item()) + " " + str(
                Y_regression_test[i][1].item()) + " " + str(Y_regression_test[i][2].item()) + " " + str(
                Y_regression_test[i][3].item()) + " " + str(Y_regression_test[i][4].item()) + " " + str(
                Y_regression_test[i][5].item())+"\n")


# 绘制分类网络的训练过程的loss和accuracy
def draw_result(loss_list, accuracy_list):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    plt.figure()
    axs[0].plot(loss_list, c="blue")
    axs[1].plot(accuracy_list, c="red")
    plt.show()


if __name__ == "__main__":
    # 创建两个网络对象
    classify_net = classify_net().to(device)
    regression_net = regression_net().to(device)

    # 加载训练集和测试集的数据
    X, Y = load_train_data("F:\\MatlabWorkspace\\nonlinear_programming\\训练集.txt")
    Y_classify, Y_regression = Y.split([3, 6], dim=1)
    X_test, Y_test = load_test_data("F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt")

    # 训练回归网络和分类网络
    train_regression(X, Y_regression, regression_net)
    loss_list, accuracy_list = train_classify(X, Y_classify, classify_net, X_test, Y_test)

    # 绘制结果，保存结果
    draw_result(loss_list, accuracy_list)
    save_result(classify_net, regression_net, X_test, Y_test, "F:\\MatlabWorkspace\\nonlinear_programming\\决策结果.txt")
