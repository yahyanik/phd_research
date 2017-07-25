import numpy as np
import cv2
#import matplotlib.pyplot as plt
from collections import deque
import imutils
import argparse
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
ap.add_argument("-v", "--video", default=0,
	help="path to the (optional) video file")
args = vars(ap.parse_args())
counter = 0
#pts = deque(maxlen=args["buffer"])
b = args["buffer"]   #gives the int value of the input command
pts = [[0 for x in range(b)] for y in range(b)]   #creating the 2 dimentional list for directions
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading cascade fiels
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml') #loading cascade fiels
#eye_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
cap = cv2.VideoCapture(0)   #loading the video file
while True:
    ret, img = cap.read()  
    blurred = cv2.GaussianBlur(img, (11, 11), 0)   #blur is useful to reduce the false positive and negatives
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)   
    direction = ""
    faces = face_cascade.detectMultiScale(gray)   #using the cascaded file
    count = 0
    for (x,y,w,h) in faces:   # for each person detected
        cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),2) #blue and with width of 2
        count = count +1
        center = [(x+x+w)/2 , (y+y+h)/2] 
#        pts.appendleft(center)
        del pts[(count-1)][-1]
        pts[(count-1)].insert(0,center) #pts is a list that needs 0 in the bebining, but counter is 1
        if counter >= 10 and pts[0][-10] != 0 :
#        if counter >= 10 and pts[-10] != 0 :
            dX = (pts[0][-10])[0] - (pts[0][0])[0]  #change the list tu small and then read the x dimention from it
            dY = (pts[0][-10])[1] - (pts[0][0])[1]
#            dX = (pts[-10])[0] - (pts[0])[0]  #change the list tu small and then read the x dimention from it
#            dY = (pts[-10])[1] - (pts[0])[1]
            (dirX, dirY) = ("", "")   #to detect the way person moves
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            else:
                direction = dirX if dirX != "" else dirY
#        roi_gray = gray[y:y+h,x:x+w]
#        roi_color = img[y:y+h,x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray)
#        for (ex, ey,ew,eh) in eyes:
#            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0), 2)
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(img,str(count),(x,y), font, 1, (255,255,0),2, cv2.LINE_AA)#avali bozorgi va adade dovom ghotr
        cv2.putText(img, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65,
                (0, 0, 255), 3)       # to print on the picture how many people are there in the frame and their direction
    cv2.imshow('img',img)
    counter = counter+1
    k = cv2.waitKey(4)
    if k == 2:
        break   
cap.release()
cv2.destroyAllWindows()


