import os
import cv2
import py360convert


class image_processor:
    def __init__(self):
        pass

    # 进程函数，从输入管道中取出任务，执行任务，将任务结果放入输出管道，不断循环
    def process_run(self, q):
        print("边缘服务器开启任务处理进程 pid=" + str(os.getpid()))

        while True:
            q_in = q[0]
            q_out = q[1]
            res = q_in.get()
            resolution = (int)(res[0])
            angle = res[1]
            img = res[2]

            if angle == "forth":
                angle = 0
            elif angle == "right":
                angle = 1
            elif angle == "back":
                angle = 2
            elif angle == "left":
                angle = 3
            elif angle == "up":
                angle = 4
            elif angle == "down":
                angle = 5

            # https://github.com/sunset1995/py360convert
            img = py360convert.e2c(img, face_w=resolution, mode='bilinear', cube_format='list')[angle]
            if angle == 1:
                img = cv2.flip(img, 1)

            q_out.put(img)
