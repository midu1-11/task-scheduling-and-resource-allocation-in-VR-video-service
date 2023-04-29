# -*- coding: utf-8 -*-
import socket  # 导入socket模块
from multiprocessing import Process, Queue
from image_process import *
import base64


class CloudServer:
    def __init__(self, q_list):
        self.server_socket = socket.socket()
        self.task_list = []
        self.decision_list = []
        self.q_list = q_list
        self.result_list = []

        self.pose_cap = cv2.VideoCapture('pose.mp4')
        self.face_cap = cv2.VideoCapture('face.mp4')
        self.hand_cap = cv2.VideoCapture('hand.mp4')
        self.video_cap = cv2.VideoCapture('video.mp4')

        self.net_set()
        self.accept()

    def net_set(self):
        host = "10.0.24.7"  # 主机IP 10.0.24.7
        port = 12345  # 端口号
        self.server_socket.bind((host, port))  # 绑定端口
        self.server_socket.listen(5)  # 设置最多连接数

    def accept(self):
        while True:
            cloud_server_sock, addr = self.server_socket.accept()  # 创建客户端连接
            data = cloud_server_sock.recv(1024).decode()  # 获取客户端请求数
            data = data.split(" ")

            for i in range(len(data)):
                one_data = data[i].split(":")
                self.task_list.append(one_data[0])
                self.decision_list.append(one_data[1])

            print(self.task_list, self.decision_list)  # 打印接收到的数据
            for decision in self.decision_list:
                if decision == "True":
                    self.result_list.append(True)
                else:
                    self.result_list.append(False)

            self.pipe_to_process()

            while self.exit_false(self.result_list):
                if not self.q_list[0][1].empty():
                    img = self.q_list[0][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'0' + base64.b64encode(base64_str)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[0] = True
                elif not self.q_list[1][1].empty():
                    img = self.q_list[1][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'1' + base64.b64encode(base64_str)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[1] = True
                elif not self.q_list[2][1].empty():
                    img = self.q_list[2][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'2' + base64.b64encode(base64_str)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    cloud_server_sock.send(base64_byte)
                    self.result_list[2] = True

            self.task_list = []
            self.decision_list = []
            self.result_list = []

    def exit_false(self, result_list):
        for result in result_list:
            if not result:
                return True
        return False

    def pipe_to_process(self):

        for i in range(len(self.result_list)):
            if self.result_list[i] == False:
                if self.task_list[i] == "pose":
                    img = self.pose_cap.read()[1]
                    self.q_list[i][0].put(("pose", img))
                elif self.task_list[i] == "face":
                    img = self.face_cap.read()[1]
                    self.q_list[i][0].put(("face", img))
                elif self.task_list[i] == "hand":
                    img = self.hand_cap.read()[1]
                    self.q_list[i][0].put(("hand", img))
                elif self.task_list[i] == "artwork":
                    img = self.video_cap.read()[1]
                    self.q_list[i][0].put(("artwork", img))
                elif self.task_list[i] == "blur":
                    img = self.video_cap.read()[1]
                    self.q_list[i][0].put(("blur", img))
                elif self.task_list[i] == "sharp":
                    img = self.video_cap.read()[1]
                    self.q_list[i][0].put(("sharp", img))


if __name__ == "__main__":
    q_list = [[Queue() for j in range(2)] for i in range(3)]

    for i in range(3):
        process = Process(target=image_processor().process_run, args=(q_list[i],))
        process.start()

    CloudServer(q_list)
