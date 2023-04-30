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
        self.resolution_list = []
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
            cloud_server_sock, addr = self.server_socket.accept()
            data = cloud_server_sock.recv(1024).decode()
            data = data.split(" ")

            for i in range(len(data)):
                one_data = data[i].split(":")
                self.task_list.append(one_data[0])
                self.decision_list.append(one_data[1])
                self.resolution_list.append(one_data[2])

            print(self.task_list, self.decision_list)  # 打印接收到的数据
            for decision in self.decision_list:
                if decision == "True":
                    self.result_list.append(True)
                else:
                    self.result_list.append(False)

            self.pipe_to_process()

            wait_time1 = 1
            wait_time2 = 2

            while self.exit_false(self.result_list):
                if not self.q_list[0][1].empty():
                    img = self.q_list[0][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'0' + base64.b64encode(base64_str)
                    print("length0="+str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[0] = True
                elif not self.q_list[1][1].empty():
                    img = self.q_list[1][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'1' + base64.b64encode(base64_str)
                    print("length1=" + str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[1] = True
                elif not self.q_list[2][1].empty():
                    img = self.q_list[2][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'2' + base64.b64encode(base64_str)
                    print("length2=" + str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[2] = True
                elif not self.q_list[3][1].empty():
                    img = self.q_list[3][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'3' + base64.b64encode(base64_str)
                    print("length3=" + str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[3] = True
                elif not self.q_list[4][1].empty():
                    img = self.q_list[4][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'4' + base64.b64encode(base64_str)
                    print("length4=" + str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[4] = True
                elif not self.q_list[5][1].empty():
                    img = self.q_list[5][1].get()
                    base64_str = cv2.imencode('.jpg', img)[1].tostring()
                    base64_byte = b'5' + base64.b64encode(base64_str)
                    print("length5=" + str(len(base64_byte)))
                    cloud_server_sock.recv(16)
                    cloud_server_sock.send((str(len(base64_byte))).encode())
                    # time.sleep(wait_time1)
                    cloud_server_sock.send(base64_byte)
                    # cloud_server_sock.recv(16)
                    # time.sleep(wait_time2)
                    self.result_list[5] = True

            self.task_list = []
            self.decision_list = []
            self.resolution_list = []
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
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("pose", img))
                elif self.task_list[i] == "face":
                    img = self.face_cap.read()[1]
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("face", img))
                elif self.task_list[i] == "hand":
                    img = self.hand_cap.read()[1]
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("hand", img))
                elif self.task_list[i] == "artwork":
                    img = self.video_cap.read()[1]
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("artwork", img))
                elif self.task_list[i] == "blur":
                    img = self.video_cap.read()[1]
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("blur", img))
                elif self.task_list[i] == "sharp":
                    img = self.video_cap.read()[1]
                    x_size = int(self.resolution_list[i].split("x")[0])
                    y_size = int(self.resolution_list[i].split("x")[1])
                    img = cv2.resize(img, (x_size, y_size))
                    self.q_list[i][0].put(("sharp", img))


if __name__ == "__main__":
    q_list = [[Queue() for j in range(2)] for i in range(6)]

    for i in range(6):
        process = Process(target=image_processor().process_run, args=(q_list[i],))
        process.start()

    CloudServer(q_list)
