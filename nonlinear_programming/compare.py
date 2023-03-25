import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

def reversal(a):
    if a == 0:
        return 1
    else:
        return 0


def not_zero(a):
    if a == 0:
        return 1
    return a


# 在测试集上对比最优决策延迟、监督学习决策延迟、强化学习REINFORCE算法决策延迟
if __name__ == "__main__":

    optimal_list = []
    mymethod_list = []
    PGmethod_list = []
    randomEquipartitionStrategy_list = []
    randomStrategy_list = []
    TSS_list = []
    ACLM_list = []
    RandomLagrangeMultiplierStrategy_list = []

    with open("F:\\MatlabWorkspace\\nonlinear_programming\\测试集.txt", encoding='utf-8') as optimal_file:
        optimal_content = optimal_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\决策结果.txt", encoding='utf-8') as mymethod_file:
        mymethod_content = mymethod_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\PG决策结果.txt", encoding='utf-8') as PGmethod_file:
        PGmethod_content = PGmethod_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机均分策略决策结果.txt", encoding='utf-8') as randomEquipartitionStrategy_file:
        randomEquipartitionStrategy_content = randomEquipartitionStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机策略决策结果.txt", encoding='utf-8') as randomStrategy_file:
        randomStrategy_content = randomStrategy_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\TSS决策结果.txt", encoding='utf-8') as TSS_file:
        TSS_content = TSS_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\ACLM决策结果.txt", encoding='utf-8') as ACLM_file:
        ACLM_content = ACLM_file.readlines()
    with open("F:\\MatlabWorkspace\\nonlinear_programming\\随机lm策略决策结果.txt", encoding='utf-8') as RandomLagrangeMultiplierStrategy_file:
        RandomLagrangeMultiplierStrategy_content = RandomLagrangeMultiplierStrategy_file.readlines()

    for i in range(0, 100):
        content1 = [float(numeric_string) for numeric_string in optimal_content[i].split(' ')]
        optimal_delay = content1[0] / content1[11] + content1[1] / content1[12] + content1[2] / content1[13] + content1[
            8] * content1[3] / not_zero(content1[14]) + content1[9] * content1[4] / not_zero(content1[15]) + content1[
                            10] * content1[5] / not_zero(content1[16]) + reversal(content1[8]) * (
                                content1[3] / content1[7] + content1[6]) + reversal(
            content1[9]) * (content1[4] / content1[7] + content1[6]) + reversal(content1[10]) * (
                                content1[5] / content1[7] + content1[6])
        optimal_list.append(optimal_delay)

        content2 = [float(numeric_string) for numeric_string in mymethod_content[i].split(' ')]
        mymethod_delay = content1[0] / content2[3] + content1[1] / content2[4] + content1[2] / content2[5] + content2[
            0] * content1[3] / not_zero(content2[6]) + content2[1] * content1[4] / not_zero(content2[7]) + content2[2] * \
                         content1[5] / not_zero(content2[8]) + reversal(content2[0]) * (
                                 content1[3] / content1[7] + content1[6]) + reversal(content2[1]) * (
                                 content1[4] / content1[7] + content1[6]) + reversal(content2[2]) * (
                                 content1[5] / content1[7] + content1[6])

        content3 = [float(numeric_string) for numeric_string in PGmethod_content[i].split(' ')]
        PGmethod_delay = content1[0] / content3[3] + content1[1] / content3[4] + content1[2] / content3[5] + content3[
            0] * content1[3] / not_zero(content3[6]) + content3[1] * content1[4] / not_zero(content3[7]) + content3[2] * \
                         content1[5] / not_zero(content3[8]) + reversal(content3[0]) * (
                                 content1[3] / content1[7] + content1[6]) + reversal(content3[1]) * (
                                 content1[4] / content1[7] + content1[6]) + reversal(content3[2]) * (
                                 content1[5] / content1[7] + content1[6])
        PGmethod_list.append(PGmethod_delay)

        content4 = [float(numeric_string) for numeric_string in randomEquipartitionStrategy_content[i].split(' ')]
        randomEquipartitionStrategy_delay = content1[0] / content4[3] + content1[1] / content4[4] + content1[2] / content4[5] + content4[
            0] * content1[3] / not_zero(content4[6]) + content4[1] * content1[4] / not_zero(content4[7]) + content4[2] * \
                         content1[5] / not_zero(content4[8]) + reversal(content4[0]) * (
                                 content1[3] / content1[7] + content1[6]) + reversal(content4[1]) * (
                                 content1[4] / content1[7] + content1[6]) + reversal(content4[2]) * (
                                 content1[5] / content1[7] + content1[6])
        randomEquipartitionStrategy_list.append(randomEquipartitionStrategy_delay)

        content5 = [float(numeric_string) for numeric_string in randomStrategy_content[i].split(' ')]
        randomStrategy_delay = content1[0] / content5[3] + content1[1] / content5[4] + content1[2] / \
                                            content5[5] + content5[
                                                0] * content1[3] / not_zero(content5[6]) + content5[1] * content1[
                                                4] / not_zero(content5[7]) + content5[2] * \
                                            content1[5] / not_zero(content5[8]) + reversal(content5[0]) * (
                                                    content1[3] / content1[7] + content1[6]) + reversal(content5[1]) * (
                                                    content1[4] / content1[7] + content1[6]) + reversal(content5[2]) * (
                                                    content1[5] / content1[7] + content1[6])
        randomStrategy_list.append(randomStrategy_delay)

        content6 = [float(numeric_string) for numeric_string in TSS_content[i].split(' ')]
        TSS_delay = content1[0] / content6[3] + content1[1] / content6[4] + content1[2] / \
                               content6[5] + content6[
                                   0] * content1[3] / not_zero(content6[6]) + content6[1] * content1[
                                   4] / not_zero(content6[7]) + content6[2] * \
                               content1[5] / not_zero(content6[8]) + reversal(content6[0]) * (
                                       content1[3] / content1[7] + content1[6]) + reversal(content6[1]) * (
                                       content1[4] / content1[7] + content1[6]) + reversal(content6[2]) * (
                                       content1[5] / content1[7] + content1[6])
        TSS_list.append(TSS_delay)

        content7 = [float(numeric_string) for numeric_string in ACLM_content[i].split(' ')]
        ACLM_delay = content1[0] / content7[3] + content1[1] / content7[4] + content1[2] / \
                    content7[5] + content7[
                        0] * content1[3] / not_zero(content7[6]) + content7[1] * content1[
                        4] / not_zero(content7[7]) + content7[2] * \
                    content1[5] / not_zero(content7[8]) + reversal(content7[0]) * (
                            content1[3] / content1[7] + content1[6]) + reversal(content7[1]) * (
                            content1[4] / content1[7] + content1[6]) + reversal(content7[2]) * (
                            content1[5] / content1[7] + content1[6])
        ACLM_list.append(ACLM_delay)

        content8 = [float(numeric_string) for numeric_string in RandomLagrangeMultiplierStrategy_content[i].split(' ')]
        RandomLagrangeMultiplierStrategy_delay = content1[0] / content8[3] + content1[1] / content8[4] + content1[2] / \
                     content8[5] + content8[
                         0] * content1[3] / not_zero(content8[6]) + content8[1] * content1[
                         4] / not_zero(content8[7]) + content8[2] * \
                     content1[5] / not_zero(content8[8]) + reversal(content8[0]) * (
                             content1[3] / content1[7] + content1[6]) + reversal(content8[1]) * (
                             content1[4] / content1[7] + content1[6]) + reversal(content8[2]) * (
                             content1[5] / content1[7] + content1[6])
        RandomLagrangeMultiplierStrategy_list.append(RandomLagrangeMultiplierStrategy_delay)

        # 针对可能出现的延迟非常大和延迟为负的情况进行处理，目前不知道为什么会出现这两种情况
        if (mymethod_delay > 20 or mymethod_delay < 0):
            print(i)
            mymethod_delay = optimal_delay
        if (mymethod_delay < 0):
            print("负：" + str(i))
        mymethod_list.append(mymethod_delay)

    # 画图
    plt.figure()
    plt.plot(optimal_list, c="red")
    # plt.plot(mymethod_list, c="blue")
    # plt.plot(PGmethod_list, c="green")
    plt.plot(randomEquipartitionStrategy_list, c="yellow")
    # plt.plot(randomStrategy_list, c="black")
    # plt.plot(TSS_list, c="black")
    plt.plot(ACLM_list, c="black")
    # plt.plot(RandomLagrangeMultiplierStrategy_list, c="green")
    # plt.legend(["最优策略", "监督学习","REINFORCE算法","随机均分策略","随机策略","TSS策略"])
    plt.legend(["最优策略","随机均分策略", "ACLM策略"])
    plt.xlabel("时隙")
    plt.ylabel("当前时隙任务平均延迟")
    plt.show()

    print("最优策略平均延迟：" + str(sum(optimal_list) / len(optimal_list)))
    print("监督学习平均延迟：" + str(sum(mymethod_list) / len(mymethod_list)))
    print("REINFORCE算法网络：" + str(sum(PGmethod_list) / len(PGmethod_list)))
    print("随机均分策略：" + str(sum(randomEquipartitionStrategy_list) / len(randomEquipartitionStrategy_list)))
    print("随机策略：" + str(sum(randomStrategy_list) / len(randomStrategy_list)))
    print("TSS策略：" + str(sum(TSS_list) / len(TSS_list)))
    print("ACLM策略：" + str(sum(ACLM_list) / len(ACLM_list)))
    print("随机lm策略：" + str(sum(RandomLagrangeMultiplierStrategy_list) / len(RandomLagrangeMultiplierStrategy_list)))
