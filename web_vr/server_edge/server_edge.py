from multiprocessing import Process, Queue, set_start_method
import time, random, os
import cv2
from image_process import *
import socket
import base64
from threading import Thread
from edge_controller import Controller
from graph_neural_network_advantage_actor_critic_lagrange_multiplier import *

NEWLINE = "\r\n"

binary_type_files = set(["jpg", "jpeg", "mp3", "png", "html", "js", "css"])


def should_return_binary(file_extension):
    """
    Returns `True` if the file with `file_extension` should be sent back as
    binary.
    """
    return file_extension in binary_type_files


def get_file_contents(file_name):
    """Returns the text content of `file_name`"""
    with open(file_name, "r") as f:
        return f.read()


class HttpServer:
    def __init__(self, q_list, host="localhost", port=9001):
        print(f"Server started. Listening at http://{host}:{port}/")
        self.host = host
        self.port = port

        self.q_list = q_list
        self.client_sock_list = []
        self.thread_list = []
        self.task_list = []
        self.resolution_list = []
        self.decision_list = [True, True, False, True, True, True]

        self.count = -1
        self.strategy = ""
        self.pose_cap = cv2.VideoCapture('pose.mp4')
        self.face_cap = cv2.VideoCapture('face.mp4')
        self.hand_cap = cv2.VideoCapture('hand.mp4')
        self.video_cap = cv2.VideoCapture('video.mp4')
        self.controller = Controller()

        self.setup_socket()
        self.accept()

        self.teardown_socket()

    def setup_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(128)

    def teardown_socket(self):
        if self.sock is not None:
            self.sock.shutdown()
            self.sock.close()

    def accept(self):
        while True:
            for i in range(6):
                (client, address) = self.sock.accept()
                self.client_sock_list.append(client)
                th = Thread(target=self.accept_request, args=(client,))
                th.start()
                self.thread_list.append(th)

            for th in self.thread_list:
                th.join()

            cloud_message_thread = Thread(target=self.cloud_to_process)
            cloud_message_thread.start()

            self.make_decision()

            print(self.task_list)
            print(self.resolution_list)
            print(self.decision_list)
            print(self.strategy)

            self.pipe_to_process()

            while not (getattr(self.client_sock_list[0], '_closed') and getattr(self.client_sock_list[1],
                                                                                '_closed') and getattr(
                self.client_sock_list[2], '_closed') and getattr(self.client_sock_list[3], '_closed') and getattr(
                self.client_sock_list[4], '_closed') and getattr(self.client_sock_list[5], '_closed')):

                if not self.q_list[0][1].empty():
                    img = self.q_list[0][1].get()
                    self.send_back(img, self.client_sock_list[0])
                elif not self.q_list[1][1].empty():
                    img = self.q_list[1][1].get()
                    self.send_back(img, self.client_sock_list[1])
                elif not self.q_list[2][1].empty():
                    img = self.q_list[2][1].get()
                    self.send_back(img, self.client_sock_list[2])
                elif not self.q_list[3][1].empty():
                    img = self.q_list[3][1].get()
                    self.send_back(img, self.client_sock_list[3])
                elif not self.q_list[4][1].empty():
                    img = self.q_list[4][1].get()
                    self.send_back(img, self.client_sock_list[4])
                elif not self.q_list[5][1].empty():
                    img = self.q_list[5][1].get()
                    self.send_back(img, self.client_sock_list[5])

            self.client_sock_list = []
            self.thread_list = []
            self.task_list = []
            self.resolution_list = []

    def make_decision(self):
        if self.strategy=="random":
            for i in range(len(self.decision_list)):
                if random.random()>0.5:
                    self.decision_list[i] = True
                else:
                    self.decision_list[i] = False
        elif self.strategy=="rl":
            self.decision_list = self.controller.get_decision_list()
        # self.decision_list = [True,True,True,True,True,True]
        # self.decision_list = [False, False, False, False, False, False]

    def accept_request(self, client_sock):
        data = client_sock.recv(4096)

        req = data.decode("utf-8")
        formatted_data = req.strip().split(NEWLINE)
        request_words = formatted_data[0].split()
        if len(request_words) == 0:
            return
        requested_file = request_words[1][1:]
        self.task_list.append((requested_file.split(".")[0]).split("|")[0])
        self.resolution_list.append((requested_file.split(".")[0]).split("|")[1])
        self.strategy = (requested_file.split(".")[0]).split("|")[2]

    def cloud_to_process(self):
        s = socket.socket()  # 创建TCP/IP套接字
        host = "43.138.30.47"  # 获取主机地址 43.138.30.47
        port = 12345  # 设置端口号
        s.connect((host, port))  # 主动初始化TCP服务连接

        send_data = ""
        for i in range(len(self.task_list)):
            send_data += self.task_list[i] + ":" + str(self.decision_list[i]) + ":" + self.resolution_list[i]
            if i != len(self.task_list) - 1:
                send_data += " "
        s.send(send_data.encode())  # 发送TCP数据

        cloud_num = 0
        for decision in self.decision_list:
            if decision == False:
                cloud_num += 1

        # data_list = []

        for i in range(cloud_num):
            s.send(b"ready")
            data_length = int(s.recv(1024).decode())
            # print("data_length "+str(data_length))

            data = b""
            while True:
                recvData = s.recv(2056)
                data += recvData
                if len(data) >= data_length:
                    break

            # data_list.append(data)
            th = Thread(target=self.send_back_from_cloud,args=(data[1:], self.client_sock_list[data[0] - 48]))
            th.start()

        # for data in data_list:
        #     self.send_back_from_cloud(data[1:], self.client_sock_list[data[0] - 48])

        s.close()

    def pipe_to_process(self):

        for i in range(len(self.decision_list)):
            if self.decision_list[i] == True:
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

    def send_back(self, img, client_sock):
        response = self.get_request(img)
        client_sock.send(response)

        client_sock.shutdown(1)
        client_sock.close()

    def send_back_from_cloud(self, content, client_sock):
        builder = ResponseBuilder()

        builder.content = b"data:image/jpeg;base64," + content

        builder.set_status("200", "OK")

        builder.add_header("Access-Control-Allow-Headers", "Content-Type")
        builder.add_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        builder.add_header("Access-Control-Allow-Origin", "*")
        builder.add_header("Content-Type", "text/plain; charset=utf-8")

        client_sock.send(builder.build())

        client_sock.shutdown(1)
        client_sock.close()

    def get_request(self, img):
        builder = ResponseBuilder()

        builder.my_set_content(img)

        builder.set_status("200", "OK")

        builder.add_header("Access-Control-Allow-Headers", "Content-Type")
        builder.add_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        builder.add_header("Access-Control-Allow-Origin", "*")
        builder.add_header("Content-Type", "text/plain; charset=utf-8")

        return builder.build()


class ResponseBuilder:
    """
    This class is here for your use if you want to use it. This follows
    the builder design pattern to assist you in forming a response. An
    example of its use is in the `method_not_allowed` function.
    Its use is optional, but it is likely to help, and completing and using
    this function to build your responses will give 5 bonus points.
    """

    def __init__(self):
        """
        Initialize the parts of a response to nothing.
        """
        self.headers = []
        self.status = None
        self.content = None

    def add_header(self, headerKey, headerValue):
        """ Adds a new header to the response """
        self.headers.append(f"{headerKey}: {headerValue}")

    def set_status(self, statusCode, statusMessage):
        """ Sets the status of the response """
        self.status = f"HTTP/1.1 {statusCode} {statusMessage}"

    def set_content(self, content):
        """ Sets `self.content` to the bytes of the content """
        if isinstance(content, (bytes, bytearray)):
            self.content = content
        else:
            self.content = content.encode("utf-8")

    def my_set_content(self, img):
        base64_str = cv2.imencode('.jpg', img)[1].tostring()
        base64_byte = base64.b64encode(base64_str)

        self.content = b"data:image/jpeg;base64," + base64_byte

    # TODO Complete the build function
    def build(self):

        response = self.status
        response += NEWLINE
        for i in self.headers:
            response += i
            response += NEWLINE
        # response += NEWLINE
        response += NEWLINE
        response = response.encode("utf-8")
        response += self.content

        return response


if __name__ == '__main__':
    q_list = [[Queue() for j in range(2)] for i in range(6)]

    for i in range(6):
        process = Process(target=image_processor().process_run, args=(q_list[i],))
        process.start()

    HttpServer(q_list)

    # cap1 = cv2.VideoCapture('face.mp4')
    # cap2 = cv2.VideoCapture('hand.mp4')
    # cap3 = cv2.VideoCapture('pose.mp4')
    #
    # while True:
    #     img1 = cap1.read()[1]
    #     img2 = cap2.read()[1]
    #     img3 = cap3.read()[1]
    #
    #     q_list[0][0].put(("face", img1))
    #     q_list[1][0].put(("hand", img2))
    #     q_list[2][0].put(("pose", img3))
    #
    #     o1 = q_list[0][1].get()
    #     o2 = q_list[1][1].get()
    #     o3 = q_list[2][1].get()
    #     cv2.imshow("1", o1)
    #     cv2.imshow("2", o2)
    #     cv2.imshow("3", o3)
    #     cv2.waitKey(1)
    #
    # print('进程间通信-队列-主进程')
