#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:13:49 2022

@author: swathiasok
"""

#Importing libraries

import cv2


#Importing cascades

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


#Detect function

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx,sy,sw,sh) in smile:
            img = cv2.rectangle(roi_frame,(sx,sy), (sx+sw,sy+sh), (255,0,0), 2)
            print("Image Saved")
            path=r'Images/img.jpg'
            cv2.imwrite(path,img)
    return frame


# Real-time detection

video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;


# Detect face in an image

#frame = cv2.imread('Images/image1.jpeg')
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#canvas = detect(gray,frame)
#path=r'Images/img.jpg'
#cv2.imwrite(path,canvas)


video_capture.release()
cv2.destroyAllWindows()