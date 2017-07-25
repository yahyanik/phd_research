import numpy as np
import cv2
#import matplotlib.pyplot as plt
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

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
    help="max buffer size")
ap.add_argument("-v", "--video", default=f2,
    help="path to the (optional) video file")
ap.add_argument("-fc", "--frame_check",type=int, default=10,
    help="in how many frames check for new person")
ap.add_argument("-fn", "--frame_noise",type=int, default=2,
    help="how many frames to detect a person to add it")
ap.add_argument("-hc", "--how_close",type=int, default=50,
    help="how many pixels should be the difference between the centers to considered the new person")
ap.add_argument("-f", "--gussian_filter",type=int, default=17,
    help="type an odd positive integer to use in gussian blur ")
args = vars(ap.parse_args())


cv2.namedWindow("tracking")
camera = cv2.VideoCapture(args["video"])
ok, image=camera.read()
#print len(image)
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



#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading cascade files
face_cascade = cv2.CascadeClassifier('haarcascade_fullbody_1.xml') #loading cascade files
#face_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml') #loading cascade files
blurred = cv2.GaussianBlur(image, (filter, filter), 0)   #blur is useful to reduce the false positive and negatives
gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY) 

faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:   # for each person detected in the first frame
#    bbxos.append((y,x,h,w))
    bbxos.append((x,y,w,h))
#    print bbxos
    if bbxos != []: #making sure that at least one person was detected
        name = str("tracker"+"_"+str(len(a)))
        a[name] = tracker(args["buffer"])
#        print a


while camera.isOpened():

    ok, image=camera.read()
    if not ok:  #breaking if there is a problem with the video
        print 'no image to read'
        break

    if not init_once and bbxos != []:   # initiate the tracking for the first frame
#        print bbxos
        i = 0
        print "first time detected"
        for track in a:
#            print a
#            print bbxos[i]
            ok = a[track].initiat(bbxos[i],image)
            i = i+1
    

#    ok = tracker.init(image, bbox1)
#        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbox2)
#        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbox3)
        init_once = True
#    cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))    
#    print len(bbxos)
    bboxs = []
    count = 1
    for track in a: #update the rectangular for each frame
        image,ok,box = a[track].update(image,count)
        if a[track].empty == 10 :
            aa,b,c,d = a[track].box
            blurred = cv2.GaussianBlur(image, (filter, filter), 0)   #blur is useful to reduce the false positive and negatives
            tike = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)[int(aa):int(aa+c),int(b):int(b+d)]
            faces_tike = face_cascade.detectMultiScale(tike)
            if a[track].emp(faces): del a[track]
#        print box
        count = count + 1
    if ticket :
        print "count" + str((count-1))
        ticket = False
#        print ok, box
#    boxes.append(box)
#    count = 0
#    for newbox in boxes:
#    print newbox
#        p1 = (int(newbox[0]), int(newbox[1]))
#        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
#        cv2.rectangle(image, p1, p2, (200,0,0))
#        cv2.putText(image,str(count),p1, font, 1, (255,255,0),2, cv2.LINE_AA)
#        count = count+1
#        if frame_counter > 1 :
#            center = ([(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2])
#            del pts[(count-1)][-1]
#            pts[(count-1)].insert(0,center) #pts is a list that needs 0 in the bebining, but counter is 1
#    if len(pts) >= 10 and pts[0][-10] != 0 :
#        if counter >= 10 and pts[-10] != 0 :
#            dX = (pts[0][-10])[0] - (pts[0][0])[0]  #change the list tu small and then read the x dimention from it
#            dY = (pts[0][-10])[1] - (pts[0][0])[1]
#            dX = (pts[-10])[0] - (pts[0])[0]  #change the list tu small and then read the x dimention from it
#            dY = (pts[-10])[1] - (pts[0])[1]
#            (dirX, dirY) = ("", "")   #to detect the way person moves
#            if np.abs(dX) > 20:
#                dirX = "East" if np.sign(dX) == 1 else "West"
#            if np.abs(dY) > 20:
#                dirY = "North" if np.sign(dY) == 1 else "South"
#            if dirX != "" and dirY != "":
#                direction = "{}-{}".format(dirY, dirX)
#            else:
#                direction = dirX if dirX != "" else dirY
              
                
#    in_box = False
#    center_tracked = []
    frame_counter = frame_counter+1
#    print len(a)
    if frame_counter >= frame_check_new_person:
        frame_counter = 1
        blurred = cv2.GaussianBlur(image, (filter, filter), 0)   #blur is useful to reduce the false positive and negatives
        gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
#        for (x,y,w,h) in faces:
#        if                               ################################################ 
#            cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))
#            frame_counter = 0
#                counter_noise = 0
#            in_box = True
#        if len(faces) > len(boxes) && ~in_box:
#        print len(a)
#        print a["tracker_0"].centered()
#    for c in a:
#        print a[c].centered()
#        print a[c].centered()
        if len(faces) > len(a):

            counter_noise = counter_noise+1
            if counter_noise >= frame_reduce_false_detection :
#                print len(a)
#                print len(faces)
                counter_noise = 1
#                for track in a:               ################################################### niaz nis tamam iena hesab shavad... tu pts has ien data bara frame ghabli
#                    p1 = [int(newbox[0]), int(newbox[1])]
#                    p2 = [int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3])]
#                    center_tracked.append([(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2])
#                    print a[track].centered()
                dummy = len(a)
                for (x,y,w,h) in faces:   
                    center_face_detect = [(x+x+w)/2 , (y+y+h)/2]
                    print center_face_detect
#                    if center_face_detect
#                    print "center_face_deect"
#                    print center_face_detect
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))
                    
                    print dummy
                    new_person = True
#                    print len(a)
                    for ii in range(0,dummy):
                        print a[track].centered()
                        print "new loop"
                        print "ii" + str(ii)
                        track = a.keys()[ii]
                        print track
                        bb = (x,y,w,h) #just for one new person detected at a time
                        
#                        print len(a)
#                        print "a" 
#                        print a[track].centered()
                        if  (-closnes < center_face_detect[0] - a[track].centered()[0]  \
                            and center_face_detect[0] - a[track].centered()[0] < closnes \
                            and -closnes < center_face_detect[1] - a[track].centered()[1] \
                            and center_face_detect[1] - a[track].centered()[1] < closnes):
#                            print len(a)
#                            continue
#                            bbx_add = (y,x,h,w)
#                            print "new person added"
#                            ticket = True
#                            print len(a)
#                        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbx_add)
#                            name = str("tracker"+"_"+str(len(a)))
#                            a[name] = tracker(args["buffer"])
#                            bb = (y,x,h,w) #just for one new person detected at a time
#                            bb = (x,y,w,h) #just for one new person detected at a time
                            new_person = False
                            break
                            
#                            ok = a[name].initiat(bb,image)
#                            break
#                            tracker_names.append(name)
                    if new_person and init_once and not len_zero:
                        name = str("tracker"+"_"+str(len(a)))
                        a[name] = tracker(args["buffer"]) 
                        new_person = True
                        ok = a[name].initiat(bb,image)
                        print "new person added"
                        ticket = True
                        
                    if not (init_once) or len_zero:
                        
                        print "in initial mode"
                        for (x,y,w,h) in faces:
#                bbx_add = (y,x,h,w)
                            bb = (x,y,w,h)
                            name = str("tracker"+"_"+str(len(a)))
                            a[name] = tracker(args["buffer"])
#                print bbx
#                print bbx_add
                            ok = a[name].initiat(bb,image)
#                            tracker_names.append(name)
                        init_once = True
                        len_zero = False
                        break
                    
    for track in a:
        if not (-10 < a[track].centered()[0] \
            and a[track].centered()[0] < len(image)+100 \
            and -10 < a[track].centered()[1] \
            and a[track].centered()[1] < len(image[0])+100):
            
#            aa= a[track].__del__()
            del a[track]
#            print aa
            
        if len(a) == 0:
            len_zero = True
            print "hala false"
#    for track in a:
#        print a[track].centered()        
            
#    print len(a)         
#    cv2.putText(image, a["tracker_0"].direction(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65,
#                (0, 0, 255), 3)       # to print on the picture how many people are there in the frame and their direction
    cv2.imshow('tracking', image)
#    print a["tracker_0"].list()
#    cv2.imshow('blurred' , blurred)
    k = cv2.waitKey(220)
    if k == 217 : break # esc pressed


camera.release()
cv2.destroyAllWindows()

#im = cv2.imread("image.jpg")
     
    # Select ROI
#r = cv2.selectROI(im)
     
    # Crop image
#imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
#cv2.imshow("Image", imCrop)
#cv2.waitKey(0)
