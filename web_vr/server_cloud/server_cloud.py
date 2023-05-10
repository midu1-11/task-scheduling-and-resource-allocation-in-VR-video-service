# -*- coding: utf-8 -*-
import socket
from multiprocessing import Process, Queue
from image_process import *
import base64


class CloudServer:
    def __init__(self, q_list):
        self.server_socket = socket.socket()
        self.task_list = []
        self.decision_list = []
        self.resolution_list = []
        self.angle_list = []
        self.q_list = q_list
        self.result_list = []

        self.animation_cap = cv2.VideoCapture('animation.mp4')
        self.dive_cap = cv2.VideoCapture('dive.mp4')
        self.dinosaur_cap = cv2.VideoCapture('dinosaur.mp4')
        self.eagle_cap = cv2.VideoCapture('eagle.mp4')
        self.Iceland_cap = cv2.VideoCapture('Iceland.mp4')
        self.zombie_cap = cv2.VideoCapture('zombie.mp4')

        # 配置网络
        self.net_set()

        # 主要业务处理函数
        self.accept()

    def net_set(self):
        host = "10.0.24.7"  # 云服务器内网ip 10.0.24.7  注意这里一定不要填成公网ip
        port = 12345  # 端口号
        self.server_socket.bind((host, port))  # 绑定端口
        self.server_socket.listen(5)  # 设置最多连接数

    def accept(self):
        while True:
            cloud_server_sock, addr = self.server_socket.accept()
            data = cloud_server_sock.recv(1024).decode()
            data = data.split(" ")

            # 解析边缘中心控制器发来的任务信息
            for i in range(len(data)):
                one_data = data[i].split(":")
                self.task_list.append(one_data[0])
                self.decision_list.append(one_data[1])
                self.resolution_list.append(one_data[2])
                self.angle_list.append(one_data[3])

            # 打印接收到的任务信息
            print(self.task_list, self.decision_list, self.resolution_list)
            for decision in self.decision_list:
                if decision == "True":
                    self.result_list.append(True)
                else:
                    self.result_list.append(False)

            # 将所有在云服务器处理的任务通过队列发送给相应的任务处理进程
            self.pipe_to_process()

            # 只有result_list中全为True(也就是所有任务结果都已发回)才会结束循环
            while self.exit_false(self.result_list):
                if not self.q_list[0][1].empty():
                    img = self.q_list[0][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    # 标明这是0#客户端的任务结果
                    base64_byte = b'0' + base64.b64encode(base64_str)
                    # 只有接收到边缘中心控制器发来的ready消息才会开始发送
                    # 这样写是因为云服务器和边缘中心控制器靠一条TCP传多个客户端的任务结果
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[0] = True
                elif not self.q_list[1][1].empty():
                    img = self.q_list[1][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'1' + base64.b64encode(base64_str)
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[1] = True
                elif not self.q_list[2][1].empty():
                    img = self.q_list[2][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'2' + base64.b64encode(base64_str)
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[2] = True
                elif not self.q_list[3][1].empty():
                    img = self.q_list[3][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'3' + base64.b64encode(base64_str)
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[3] = True
                elif not self.q_list[4][1].empty():
                    img = self.q_list[4][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'4' + base64.b64encode(base64_str)
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[4] = True
                elif not self.q_list[5][1].empty():
                    img = self.q_list[5][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'5' + base64.b64encode(base64_str)
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[5] = True

            self.task_list = []
            self.decision_list = []
            self.resolution_list = []
            self.angle_list = []
            self.result_list = []

    def exit_false(self, result_list):
        for result in result_list:
            if not result:
                return True
        return False

    # 这里云服务器和边缘服务器一样操作
    def pipe_to_process(self):

        for i in range(len(self.result_list)):
            if self.result_list[i] == False:
                if self.task_list[i] == "animation":
                    img = self.animation_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))
                elif self.task_list[i] == "dive":
                    img = self.dive_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))
                elif self.task_list[i] == "dinosaur":
                    img = self.dinosaur_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))
                elif self.task_list[i] == "eagle":
                    img = self.eagle_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))
                elif self.task_list[i] == "Iceland":
                    img = self.Iceland_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))
                elif self.task_list[i] == "zombie":
                    img = self.zombie_cap.read()[1]
                    self.q_list[i][0].put((self.resolution_list[i].split("x")[1], self.angle_list[i], img))


if __name__ == "__main__":
    # 每一个任务处理进程对应一个输入队列、一个输出队列
    q_list = [[Queue() for j in range(2)] for i in range(6)]

    # 创建6个任务处理进程
    for i in range(6):
        process = Process(target=image_processor().process_run, args=(q_list[i],))
        process.start()

    CloudServer(q_list)
