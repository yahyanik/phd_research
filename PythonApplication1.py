import numpy as np
import cv2
#import matplotlib.pyplot as plt
from collections import deque
#import imutils
import argparse
import sys


#img = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)       # this one is 0
#imread_color = 1
#imread_unchanged = -1

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
  
#plt.imshow(img,cmap='gray',interpolation = 'bicubic')
#plt.plot([50,100],[80,100],'r','g',linewidth=5)
#plt.show()

'''
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
print pts
cv2.polylines(img,[pts],False, (0,0,255),3)
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
cv2.putText(img,'opencv',(0,130), font, 4, (255,0,0),8, cv2.LINE_AA)#avali bozorgi va adade dovom ghotr
#cv2.line(img, (0,0), (150,150), (100,0,0),15)  #BGR
#cv2.rectangle(img, (15,25),(200,150),(0,255,0),5)
#cv2.circle(img, (15,25),55,(0,255,0),-1)
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#img = cv2.imread('image.jpg') #load image
#px = img[55,55]
#print px
#roi = img[100:150,100:150] = [255,255,255]
#w_f = img[37:111,107:194]
#img[0:74, 0:87] = w_f  #copy kardane ghesmate w_f be ien mokhtasate jadid
#cv2.imshow('',img)          
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''
img1 = cv2.imread('image2.jpg')
img2 = cv2.imread('image.jpg')
#add = cv2.add(img1,img2)
#w = cv2.addWeighted(img1,0.6,img1,0.4,0)
rows,cols,channels = img2.shape
roi = img1[0:rows,0:cols]
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray,220,255,cv2.THRESH_TOZERO_INV)   #az adade aval balatar ro siah mikone
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
img2_fg = cv2.bitwise_and(img2, img2 , mask = mask)
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols]  = dst
cv2.imshow('add',dst)          
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#img = cv2.imread('image.jpg')
#img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, tresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#gaus = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
#cv2.imshow('add',gaus)          
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#############################################################################
#############################################################################
'''
#cap = cv2.VideoCapture('output.avi')   #load video
cap = cv2.VideoCapture(0)            #shomare webcam 0 ast
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0 ,(640,480))
while True :
     ret, frame = cap.read()
     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #out.write(frame)

     cv2.imshow('fra',frame)
     cv2.imshow('gr',gray)

     if cv2.waitKey(1) & 0xff == ord('q'):
         break
cap.release()
OUT.release()
cv2.destroyAllWindows()
'''
'''
vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red= np.array([150,55,0])
    upper_red= np.array([180,150,255])
    mask = cv2.inRange(hsv,lower_red,upper_red)
    res = cv2.bitwise_and(frame,frame,mask = mask)
    #kernel = np.ones((15,15),np.float32)/255 #255 = 15*15
    kernel1 = np.ones((5,5),np.uint8)
    erosion= cv2.erode(mask ,kernel1,iterations = 1)
    dilation= cv2.dilate(mask,kernel1, iterations=1)
    #opening is to remove faulse positive and closing is to remove faulse negatives
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel1)
    #smooth = cv2.filter2D(res,-1,kernel)
    #blur = cv2.GaussianBlur(res,(15,15),0)
    #median = cv2.medianBlur(res,15)
    #cv2.imshow('frame',frame)
    cv2.imshow('fram2e',res)
    #cv2.imshow('opening',opening)
    #cv2.imshow('closing',closing)
    k = cv2.waitKey(5) &0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
'''
#edge
'''vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    laplation = cv2.Laplacian(frame,cv2.CV_64F) 
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0, ksize = 5)#be halate amoodie tasvir
    sobelx1 = cv2.Sobel(frame,cv2.CV_64F,0,1, ksize = 5)  
    edge = cv2.Canny(frame,150,100)
    cv2.imshow('original',frame)
    #cv2.imshow('laplation',laplation)
    #cv2.imshow('sobel',sobelx)
    cv2.imshow('edg',edge)
    k = cv2.waitKey(5) &0xff
    if k == 22:
        break
cv2.destroyAllWindows()
cap.release()
'''
'''
#motion detection
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2 ()
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    kar = cv2.bitwise_and(frame,frame,mask = fgmask)
    cv2.imshow('original', frame)
    cv2.imshow('fg', kar)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows() 
'''




'''



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

'''





#if len(sys.argv) != 2:
#    print('Input video name is missing')
#    exit()

#print('Select 3 tracking targets')
a = {}
tracker_names = []
bbxos = []
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
ap.add_argument("-v", "--video", default=0,
	help="path to the (optional) video file")
args = vars(ap.parse_args())
cv2.namedWindow("tracking")
#camera = cv2.VideoCapture(sys.argv[1])
camera = cv2.VideoCapture(0)

init_once = False
bbx = []
ok, image=camera.read()
#if not ok:
#    print('Failed to read video')
#    exit()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #loading cascade fiels
blurred = cv2.GaussianBlur(image, (11, 11), 0)   #blur is useful to reduce the false positive and negatives
faces = face_cascade.detectMultiScale(blurred)
for (x,y,w,h) in faces:   # for each person detected
    bbxos.append((y,x,h,w))
if bbxos != []:
    tracker = cv2.Tracker_create('KCF')
    name = str(tracker)
    a[name] = tracker
    tracker_names.append(name)
#bbox1 = cv2.selectROI('tracking', image)
#bbox2 = cv2.selectROI('tracking', image)
#bbox3 = cv2.selectROI('tracking', image)

frame_counter = 0
counter_noise = 0
frame_check_new_person = 5
frame_reduce_false_detection = 5
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
#Change_in_list = False
b = args["buffer"]   #gives the int value of the input command
pts = [[0 for x in range(b)] for y in range(b)]   #creating the 2 dimentional list for directions
#while camera.isOpened():
while True :

    center = []
    ok, image=camera.read()
    if not ok:
        print 'no image to read'
        break

    if not init_once and bbxos != []:####################################################
	print bbxos
        for bbx_add in bbxos:
	    name = str("tracker"+"_"+str(len(a)))
	    a[name] = cv2.Tracker_create('KCF')
#	    print bbx
#	    print bbx_add
	    ok = a[name].init(image, bbx_add)
	    tracker_names.append(name)

#	ok = tracker.init(image, bbox1)
#        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbox2)
#        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbox3)
        init_once = True
#    print len(bbxos)
    boxes = []
    for name in tracker_names:
    	ok, box = a[name].update(image)
#    	print ok, box
	boxes.append(box)
    count = 0
    for newbox in boxes:
#	print newbox
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, (200,0,0))
        cv2.putText(image,str(count),p1, font, 1, (255,255,0),2, cv2.LINE_AA)
        count = count+1
        if frame_counter > 1 :
            center = ([(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2])
            del pts[(count-1)][-1]
            pts[(count-1)].insert(0,center) #pts is a list that needs 0 in the bebining, but counter is 1
    if count >= 10 and pts[0][-10] != 0 :
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
    in_box = False
    center_tracked = []
    frame_counter = frame_counter+1
    if frame_counter >= frame_check_new_person:
	frame_counter = 1
	blurred = cv2.GaussianBlur(image, (11, 11), 0)   #blur is useful to reduce the false positive and negatives
        faces = face_cascade.detectMultiScale(blurred)
	for (x,y,w,h) in faces:
#	    if                               ################################################ 
	    cv2.rectangle(image, (x,y), (x+w,y+h), (0,200,0))
#	    	frame_counter = 0
#            	counter_noise = 0
#	    	in_box = True
#        if len(faces) > len(boxes) && ~in_box:
	if len(faces) > len(boxes):
	    print len(boxes)
	    print len(faces)
            counter_noise = counter_noise+1
            if counter_noise >= frame_reduce_false_detection :
                
                counter_noise = 1
                for newbox in boxes:               ################################################### niaz nis tamam iena hesab shavad... tu pts has ien data bara frame ghabli
                    p1 = [int(newbox[0]), int(newbox[1])]
                    p2 = [int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3])]
                    center_tracked.append([(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2])
		    print center_tracked
                for bbx_add in faces:   
                    center_face_detect = [(x+x+w)/2 , (y+y+h)/2]
		    print center_face_detect
		    for have in center_tracked:
                    	if -30 < center_face_detect[0] - have[0]  and center_face_detect[0] - have[0] < 30 and -30 < center_face_detect[1] - have[1] and center_face_detect[1] - have[1] < 30:
#                        	bbx_add = (y,x,h,w)
#                        ok = tracker.add(cv2.TrackerMIL_create('KCF'), image, bbx_add)
				name = str("tracker"+"_"+str(len(a)))
				a[name] = cv2.Tracker_create('KCF')
				ok = a[name].init(image, bbx_add)
				tracker_names.append(name)
		    if not init_once:
			for bbx_add in faces:
#			    bbx_add = (y,x,h,w)
			    name = str("tracker"+"_"+str(len(a)))
			    a[name] = cv2.Tracker_create('KCF')
#			    print bbx
#			    print bbx_add
			    ok = a[name].init(image, bbx_add)
			    tracker_names.append(name)
		    	init_once = True
		    	break
				
	

    cv2.imshow('tracking', image)
    k = cv2.waitKey(30)
    if k == 27 : break # esc pressed


cap.release()
cv2.destroyAllWindows()

#im = cv2.imread("image.jpg")
     
    # Select ROI
#r = cv2.selectROI(im)
     
    # Crop image
#imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
#cv2.imshow("Image", imCrop)
#cv2.waitKey(0)
