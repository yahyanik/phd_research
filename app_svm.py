import numpy as np
import cv2
from KCF_tracker import tracker
from Different_person import find_person
from collections import OrderedDict
import imutils
import argparse
import sys
import os


f1 = "EC-Main-Entrance-2017-05-21_14h20min25s670ms.mp4"
f0 = 0
f2 = "EC-Main-Entrance-2017-05-21_02h10min05s000ms.mp4"
f3 = "vtest.avi"

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
ap.add_argument("-v", "--video", default=f1,
    help="path to the (optional) video file")
ap.add_argument("-fc", "--frame_check",type=int, default=1,
    help="in how many frames check for new person")
ap.add_argument("-fn", "--frame_noise",type=int, default=1,
    help="how many frames to detect a person to add it")
ap.add_argument("-hc", "--how_close",type=int, default=5,
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

HOGCascade = cv2.HOGDescriptor()    
HOGCascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

blurred = cv2.GaussianBlur(image, (filter, filter), 0)   #blur is useful to reduce the false positive and negatives
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY) 
(faces, weights)= HOGCascade.detectMultiScale(gray,winStride=(8,8),padding=(128,128),scale=1.05)
for (x,y,w,h) in faces:   # for each person detected in the first frame
    bbxos.append((x,y,w,h))
    if bbxos != []: #making sure that at least one person was detected
        name = str("tracker"+"_"+str(len(a)))
        a[name] = tracker(args["buffer"])
        
        
        
while camera.isOpened():

    ok, image=camera.read()
    if not ok:  #breaking if there is a problem with the video
        print 'no image to read'
        break

    if not init_once and bbxos != []:   # initiate the tracking for the first frame
        i = 0
        print "first time detected"
        for track in a:
            ok = a[track].initiat(bbxos[i],image)
            i = i+1
        init_once = True
        
    bboxs = []
    count = 1
    for track in a: #update the rectangular for each frame
        image,ok,box = a[track].update(image,count,filter)
        okk = a[track].emp()
        if not okk : 
#            del a[track] 
            print"have no movement"
#            break
        count = count + 1
    if ticket :
        print "count" + str((count-1))
        ticket = False
    if len(a) == 0:
        len_zero = True

    frame_counter = frame_counter+1
    if frame_counter >= frame_check_new_person:
        frame_counter = 1
        blurred = cv2.GaussianBlur(image, (filter, filter), 0)   #blur is useful to reduce the false positive and negatives
        gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        (faces, weights)= HOGCascade.detectMultiScale(gray,0,winStride=(8,8),padding=(128,128),scale=1.05)
        
        if len(faces) > len(a):
            counter_noise = counter_noise+1
            if counter_noise >= frame_reduce_false_detection :
                counter_noise = 1
                dummy = len(a)
                for (x,y,w,h) in faces:   
                    center_face_detect = [(x+x+w)/2 , (y+y+h)/2]
                    print "center_face_detect"+str(center_face_detect)
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))
                    print "len(a)"+str(dummy)
                    new_person = True
                    for ii in range(0,dummy):
                        bb = (x,y,w,h) #just for one new person detected at a time
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
                    
#    for track in a:
#        if not (-10 < a[track].centered()[0] \
#            and a[track].centered()[0] < len(image)+100 \
#            and -10 < a[track].centered()[1] \
#            and a[track].centered()[1] < len(image[0])+100):
#            del a[track]

    cv2.imshow('tracking', image)
    k = cv2.waitKey(200)
    if k == 17 : break # esc pressed


camera.release()
cv2.destroyAllWindows()
