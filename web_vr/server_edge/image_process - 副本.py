import cv2
from cvzone.PoseModule import PoseDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import time


class image_processor:
    def __init__(self):
        self.poseDetector = PoseDetector()
        self.faceMeshDetector = FaceMeshDetector(maxFaces=1)
        self.handDetector = HandDetector(detectionCon=0.8, maxHands=2)

    def process_one_img(self,img,type):
        if type=="pose":
            return self.poseDetector.findPose(img)
        elif type=="face":
            return self.faceMeshDetector.findFaceMesh(img)[0]
        elif type=="hand":
            return self.handDetector.findHands(img)[1]
        elif type=="artwork":
            return img
        elif type=="blur":
            return cv2.blur(img, (25, 25))
        elif type=="sharp":
            return cv2.convertScaleAbs(cv2.Sobel(img, cv2.CV_32F, 1, 0))

