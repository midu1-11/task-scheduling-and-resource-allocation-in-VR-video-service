from multiprocessing import Process, Queue, set_start_method
import time, random, os
import cv2
from image_process import *
import socket
import base64
from threading import Thread

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

        self.count = -1
        self.pose_cap = cv2.VideoCapture('pose.mp4')
        self.face_cap = cv2.VideoCapture('face.mp4')
        self.hand_cap = cv2.VideoCapture('hand.mp4')
        self.video_cap = cv2.VideoCapture('video.mp4')

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
            for i in range(3):
                (client, address) = self.sock.accept()
                self.client_sock_list.append(client)
                th = Thread(target=self.accept_request, args=(client, address))
                th.start()
                self.thread_list.append(th)
                # self.accept_request(client, address)

            for th in self.thread_list:
                th.join()

            while not (getattr(self.client_sock_list[0], '_closed') and getattr(self.client_sock_list[1], '_closed') and getattr(self.client_sock_list[2], '_closed')):

                if not self.q_list[0][1].empty():
                    img = self.q_list[0][1].get()
                    self.send_back(img, self.client_sock_list[0])
                    # del self.client_sock_list[0]
                elif not self.q_list[1][1].empty():
                    img = self.q_list[1][1].get()
                    self.send_back(img, self.client_sock_list[1])
                    # del self.client_sock_list[1]
                elif not self.q_list[2][1].empty():
                    img = self.q_list[2][1].get()
                    self.send_back(img, self.client_sock_list[2])
                    # del self.client_sock_list[2]

            self.client_sock_list = []
            self.thread_list = []








    def accept_request(self, client_sock, client_addr):
        data = client_sock.recv(4096)
        req = data.decode("utf-8")

        self.pipe_to_process(req)

        # start_time = time.time()

        # response = self.process_response(req)
        # client_sock.send(response)
        #
        # # clean up
        # client_sock.shutdown(1)
        # client_sock.close()
        #
        # end_time = time.time()
        # interval = end_time - start_time
        # print("time:" + str(interval))

    def pipe_to_process(self, request):
        formatted_data = request.strip().split(NEWLINE)
        request_words = formatted_data[0].split()
        if len(request_words) == 0:
            return
        requested_file = request_words[1][1:]

        self.count += 1
        self.count %= 3

        # for i in range(3):
        #     if requested_file[i] == "1":
        #         img = self.pose_cap.read()[1]
        #         self.q_list[i][0].put(("pose", img))
        #     elif requested_file[i] == "2":
        #         img = self.face_cap.read()[1]
        #         self.q_list[i][0].put(("face", img))
        #     elif requested_file[i] == "3":
        #         img = self.hand_cap.read()[1]
        #         self.q_list[i][0].put(("hand", img))
        #     elif requested_file[i] == "4":
        #         img = self.video_cap.read()[1]
        #         self.q_list[i][0].put(("artwork", img))
        #     elif requested_file[i] == "5":
        #         img = self.video_cap.read()[1]
        #         self.q_list[i][0].put(("blur", img))
        #     elif requested_file[i] == "6":
        #         img = self.video_cap.read()[1]
        #         self.q_list[i][0].put(("sharp", img))

        if requested_file == "pose.html":
            img = self.pose_cap.read()[1]
            self.q_list[self.count][0].put(("pose", img))
        elif requested_file == "face.html":
            img = self.face_cap.read()[1]
            self.q_list[self.count][0].put(("face", img))
        elif requested_file == "hand.html":
            img = self.hand_cap.read()[1]
            self.q_list[self.count][0].put(("hand", img))
        elif requested_file == "artwork.html":
            img = self.video_cap.read()[1]
            self.q_list[self.count][0].put(("artwork", img))
        elif requested_file == "blur.html":
            img = self.video_cap.read()[1]
            self.q_list[self.count][0].put(("blur", img))
        elif requested_file == "sharp.html":
            img = self.video_cap.read()[1]
            self.q_list[self.count][0].put(("sharp", img))


    def send_back(self, img,client_sock):
        response = self.get_request(img)
        client_sock.send(response)

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
    q_list = [[Queue() for j in range(2)] for i in range(3)]

    for i in range(3):
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
