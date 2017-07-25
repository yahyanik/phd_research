import numpy as np
import cv2
from KCF_tracker import tracker
from Different_person import find_person
from collections import OrderedDict
import imutils
import argparse
import sys
import os
import argparse
import datetime
import time


f1 = "EC-Main-Entrance-2017-05-21_14h20min25s670ms.mp4"
f0 = 0
f2 = "EC-Main-Entrance-2017-05-21_02h10min05s000ms.mp4"
f3 = "vtest.avi"

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
ap.add_argument("-v", "--video", default=f1,
    help="path to the (optional) video file")
ap.add_argument("-fc", "--frame_check",type=int, default=10,
    help="in how many frames check for new person")
ap.add_argument("-fn", "--frame_noise",type=int, default=2,
    help="how many frames to detect a person to add it")
ap.add_argument("-hc", "--how_close",type=int, default=50,
    help="how many pixels should be the difference between the centers to considered the new person")
ap.add_argument("-f", "--gussian_filter",type=int, default=21,
    help="type an odd positive integer to use in gussian blur ")
args = vars(ap.parse_args())


cv2.namedWindow("tracking")
camera = cv2.VideoCapture(args["video"])
ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

how_small = 60
cap = cv2.VideoCapture("vtest.avi")
fgbg = cv2.createBackgroundSubtractorMOG2 ()
init_once = False
a=OrderedDict()  #this is a dictionary to have all of the object names
# direct to store the order of the people detected
bbx = []
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
bbxos = []
frame_counter = 0
len_zero = False
counter_noise = 0
ticket = True
filter = args["gussian_filter"]
new_person = False
frame_check_new_person = args["frame_check"]
frame_reduce_false_detection = args["frame_noise"]
closnes = args["how_close"]
not_moving = 0
faces = []
       
        
while camera.isOpened():

    ok, image=camera.read()
    if not ok:  #breaking if there is a problem with the video
        print 'no image to read'
        break

    if not (init_once):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgmask = fgbg.apply(gray)
        (im2,cnts, hierarchy) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            (x, y, w, h)= bb=cv2.boundingRect(c)    # if the contour is too small, ignore it
            if w < how_small or h <how_small:
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            name = str("tracker"+"_"+str(len(a)))
#            a[name] = tracker(args["buffer"])
#            ok = a[name].initiat(bb,image)
#            init_once = True
            
    
    frame_counter = frame_counter+1
    if (frame_counter >= frame_check_new_person) or not init_once:
        frame_counter = 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgmask = fgbg.apply(gray)
        (im2,cnts, hierarchy) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        dummy = len(a)
        for c in cnts:
            (x, y, w, h)= cv2.boundingRect(c)    # if the contour is too small, ignore it
            if w < how_small or h <how_small:
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)   
            center_face_detect = [(x+x+w)/2 , (y+y+h)/2]
            
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))
            print "len(a)"+str(dummy)
            new_person = True
            if len(faces) > len(a):
                print "center_face_detect"+str(center_face_detect)
                for ii in range(0,dummy):
                    track = a.keys()[ii]
                    print "a.center"+str(a[track].centered())
#                        print "new loop"                        
                    bb = (x,y,w,h) #just for one new person detected at a time
                    if  (-closnes < center_face_detect[0] - a[track].centered()[0]  \
                        and center_face_detect[0] - a[track].centered()[0] < closnes \
                        and -closnes < center_face_detect[1] - a[track].centered()[1] \
                        and center_face_detect[1] - a[track].centered()[1] < closnes):
                        new_person = False
                        break
                        
                if new_person and init_once and not len_zero:
                    name = str("tracker"+"_"+str(len(a)))
                    a[name] = tracker(args["buffer"]) 
                    new_person = True
                    ok = a[name].initiat(bb,image)
                    print "new person added"
                    ticket = True    
                                            
            if not (init_once) or len_zero:
                for (x,y,w,h) in faces:
                    bb = (x,y,w,h)
                    name = str("tracker"+"_"+str(len(a)))
                    a[name] = tracker(args["buffer"])
                    print "in initial mode"
                    ok = a[name].initiat(bb,image)
                init_once = True
                len_zero = False
#                    break

                    
    bboxs = []
    count = 1
    for track in a: #update the rectangular for each frame
        image,ok,box = a[track].update(image,count,filter)
        okk = a[track].emp()
        if not okk : 
            del a[track] 
            print"have no movement"

        count = count + 1
    if ticket :
        print "count" + str((count-1))
        ticket = False
            
    if len(a) == 0:
        len_zero = True

    cv2.imshow('tracking', image)
    k = cv2.waitKey(10)
    if k == 217 : break # esc pressed
#    if k & 0xFF == ord('q'): break


camera.release()
cv2.destroyAllWindows()
