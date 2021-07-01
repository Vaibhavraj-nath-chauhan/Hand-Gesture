#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
class FaceDetector():
    def __init__(self,minDetectionConfidance=.5):
        self.minDetectionConfidance = minDetectionConfidance
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidance)
    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs= []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                h,w,c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                landmarks = detection.location_data.relative_keypoints
                bboxs.append([id,bbox,detection.score,landmarks])
                if draw:
                    cv2.rectangle(img,bbox,(255,0,255),2)
                    cv2.putText(img,f"{int(detection.score[0]*100)}%",
                            (bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,
                            .4,(255,0,255),1)
        return img,bboxs
        

