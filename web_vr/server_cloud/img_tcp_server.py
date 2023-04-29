import cv2
import base64
import socket

server_socket = socket.socket()
host = "10.0.24.7"  # 主机IP 10.0.24.7
port = 11122  # 端口号
server_socket.bind((host, port))  # 绑定端口
server_socket.listen(5)  # 设置最多连接数

cloud_server_sock, addr = server_socket.accept()  # 创建客户端连接

video_cap = cv2.VideoCapture('video.mp4')

img = video_cap.read()[1]

base64_str = cv2.imencode('.jpg', img)[1].tostring()
base64_byte = base64.b64encode(base64_str)
cloud_server_sock.send((str(len(base64_byte))).encode())
cloud_server_sock.send(base64_byte)