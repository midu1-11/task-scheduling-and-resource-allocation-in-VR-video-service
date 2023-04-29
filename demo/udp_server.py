#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
udp通信例程：udp server端，修改udp_addr元组里面的ip地址，即可实现与目标机器的通信，
此处以单机通信示例，ip为127.0.0.1，实际多机通信，此处应设置为目标客户端ip地址
"""
__author__ = "River.Yang"
__date__ = "2021/4/30"
__version__ = "1.0.0"

from time import sleep
import socket


def main():
    # udp 通信地址，IP+端口号
    udp_addr = ('10.0.24.7', 9999)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定端口
    udp_socket.bind(udp_addr)

    # 等待接收对方发送的数据
    while True:
        recv_data = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数
        # 打印接收到的数据
        print("[From %s:%d]:%s" % (recv_data[1][0], recv_data[1][1], recv_data[0].decode("utf-8")))

if __name__ == '__main__':
    print("当前版本： ", __version__)
    print("udp server ")
    main()
