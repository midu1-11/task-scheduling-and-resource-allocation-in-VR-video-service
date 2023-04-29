# -*- coding: utf-8 -*-
import socket  # 导入socket模块
import base64
import cv2

s = socket.socket()  # 创建TCP/IP套接字
host = "43.138.30.47"  # 获取主机地址 43.138.30.47
port = 11122  # 设置端口号
s.connect((host, port))  # 主动初始化TCP服务连接


while True:
    data = b""
    data_length = int(s.recv(1024).decode())

    while True:
        recvData = s.recv(1024)
        data += recvData
        if len(data)==data_length:
            break
    print(" ")
# recvData = base64.b64decode(recvData)
# recvData = cv2.imdecode(recvData,1)
# print("接收到的数据为：", recvData)
# 关闭套接字
s.close()