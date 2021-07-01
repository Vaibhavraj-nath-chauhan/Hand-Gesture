#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
class poseDetector():                      
    def __init__(self,mode=False,upBody=1,smooth=True,detectionCon=0.5,trackCon=0.5):    #Default paramets which are are part of pose function
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpPose = mp.solutions.pose                                       #Calling pose function
        self.pose = self.mpPose.Pose(self.mode,                               #passing default paraments
                                     self.upBody,self.smooth,
                                     self.detectionCon,
                                     self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils                              #calling drawing function
     
    def findPose(self,img,draw=True):                                         #creating a function which will find landmarks and draw them
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)                          #coverting our img to RGB as mediapipe only take RGB    
        self.results = self.pose.process(imgRGB)                              #processing our image for landmakrds
        if self.results.pose_landmarks:                                       #if we get the landmars we draw it 
            if draw:                                                                                   
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,       #this will dwar points in connected manner 
                                           self.mpPose.POSE_CONNECTIONS)
        return img                                                            #returning our image
     #there are total of 32 landmarks which will be fetched by our module
    def getPosition(self,img,draw=True):                                      #Now we will fetch the lanmrks for further use 
        lmList=[]                                                             #creating a list to fetch all the landmarks
        if self.results.pose_landmarks:
        
            for id,lm in enumerate(self.results.pose_landmarks.landmark):     #looping landmarks list
                h,w,c = img.shape                                         
                cx,cy = int(lm.x*w) , int(lm.y*h)                             #those ladnmakrs are in decimel hence we are conveting it into image shape points
                lmList.append([id,cx,cy])                                     #append it to our list with given landmarks position as it starts with 0 to 31 
                if draw:                                                      #if we want to check we can give it a try
                    cv2.circle(img,(cx,cy),2,(255,0,0),cv2.FILLED)            
 
       return lmList                                                          #returning list of landmarks

