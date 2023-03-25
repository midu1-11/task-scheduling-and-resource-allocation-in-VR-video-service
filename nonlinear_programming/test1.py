import math
import time
import torch
import torch.nn as nn

# start_time = time.time()    # 程序开始时间
# for i in range(pow(2,1)):
#     for j in range(50):
#         a = math.sqrt(j)
#
# end_time = time.time()    # 程序结束时间
# run_time = end_time - start_time    # 程序的运行时间，单位为秒
# print(run_time)

# a=nn.Linear(2,10)
# print(1)
# import torch
# a = torch.tensor([[ 0.0349,  0.0670, -0.0612, 0.0280, -0.0222,  0.0422],
#          [-1.6719,  0.1242, -0.6488, 0.3313, -1.3965, -0.0682],
#          [-1.3419,  0.4485, -0.6589, 0.1420, -0.3260, -0.4795],
#          [-0.0658, -0.1490, -0.1684, 0.7188,  0.3129, -0.1116],
#          [-0.2098, -0.2980,  0.1126, 0.9666, -0.0178,  0.1222],
#          [ 0.1179, -0.4622, -0.2112, 1.1151,  0.1846,  0.4283]])
# # 查看tensor的维度信息：torch.Size([6, 6])
# print(a.shape[0])

# a = torch.cat([torch.tensor([[1,2]]),torch.tensor([[3,4]])],1)
# print(a)
# b=torch.tensor([[1,2]])
# print(b.shape)
# tuple1 = ([1,2],[3,4])
# print(tuple1[0],tuple1[1])
# tuple1[0][0]=0
# print(tuple1[0],tuple1[1])
# tuple1[0].append(1)
# print(tuple1[0],tuple1[1])
a = [1,2]+[0]
print(a)