import os
import cv2
from cvzone.PoseModule import PoseDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector


class image_processor:
    def __init__(self):
        pass

    def process_run(self,q):
        print("云服务器开启任务处理进程 pid="+str(os.getpid()))

        while True:
            q_in = q[0]
            q_out = q[1]
            res = q_in.get()
            type = res[0]
            img = res[1]

            if type == "pose":
                q_out.put(PoseDetector().findPose(img))
            elif type == "face":
                q_out.put(FaceMeshDetector(maxFaces=1).findFaceMesh(img)[0])
            elif type == "hand":
                q_out.put(HandDetector(detectionCon=0.8, maxHands=2).findHands(img)[1])
            elif type == "artwork":
                q_out.put(img)
            elif type == "blur":
                q_out.put(cv2.blur(img, (25, 25)))
            elif type == "sharp":
                q_out.put(cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 1, 0)))

