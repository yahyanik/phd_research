import numpy as np
import cv2
#import matplotlib.pyplot as plt
#from KCF_tracker import tracker
#from collections import OrderedDict
#import imutils
import argparse
import sys
import os


'''
a=OrderedDict()

a["name1"] = 5
a["name2"] = 6
a["name3"] = 43
print a


del a["name2"]

print a
print len(a)
a["amir"] = 546
b = sorted(a.keys())
print a
count = len(a)
for k in a :
    print a[k]
    print count
    count  = count -1
    
'''



cap = cv2.VideoCapture("EC-Main-Entrance-2017-05-21_02h10min05s000ms.mp4")
#firstFrame = None
fgbg = cv2.createBackgroundSubtractorMOG2 ()
#HOGCascade = cv2.HOGDescriptor()    
#HOGCascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
#    if firstFrame is None:
#        firstFrame = gray
#        continue
    
#    frameDelta = cv2.absdiff(firstFrame, gray)
#    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
#    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]])
    for c in contours:
        # if the contour is too small, ignore it
        (x, y, w, h) = cv2.boundingRect(c)
        zz = 40
        if w < zz or h <zz:
            continue
 
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
#        rect = cv2.minAreaRect(c)
#        box = cv2.boxPoints(rect)
#        box = np.int0(box)
#        cv2.drawContours(frame,[box],0,(0,0,255),2)
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#    for ((x, y, w, h)) in hull:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
#    cnt = contours[0]
#    rect = cv2.minAreaRect(cnt)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    x,y,w,h = cv2.boundingRect(cnt)
#    print x,y,w,h
#    x,y,w,h = cv2.boundingRect(cnt)
#    cv2.drawContours(frame,contours,-1,(0,0,255),2)
#    epsilon = 0.1*cv2.arcLength(cnt,True)
#    approx = cv2.approxPolyDP(cnt,epsilon,True)
#    (faces, weights)= HOGCascade.detectMultiScale(frame,0,winStride=(8,8),padding=(128,128),scale=1.05)
#    for (x,y,w,h) in faces:
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('original', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 427:
        break
cap.release()
cv2.destroyAllWindows() 








