import numpy as np
import cv2
from collections import deque
import imutils
import argparse
import sys
import math




class tracker:
    
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    
    # tedad is the list size to save the center history 
    def __init__ (self,tedad):  
        self.create = cv2.Tracker_create('MIL') # create the tracker
        self.pts = deque(maxlen=tedad)          #list for each object movement history
        self.box = 0                      #for the direction they are going
        self.center = []                         #for the center of the rectangular in each pixel
        self.movement = 0
        
    def initiat (self,bbx_add,image):
        ok = self.create.init(image, bbx_add)   #start tracking
        self.center = bbx_add
        return ok
    
    def update (self,image,count,filter = 11):
        ok, box = self.create.update(image)     #update for each frame
        self.box = box
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        self.center = [(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2]
        self.pts.appendleft(self.center)
        cv2.rectangle(image, p1, p2, (200,0,0))
        cv2.putText(image,str(count),p1, self.font, 1, (255,255,0),2, cv2.LINE_AA)
        return (image,ok,p1)
    
#    def __del__(self):
#        return 1

    def centered (self):
        return self.center
    
    def direction (self):
        pts = self.pts
        if len(pts) >= 10 and pts[-10] != 0 :
#        if counter >= 10 and pts[-10] != 0 :
            dX = (pts[-10])[0] - (pts[0])[0]  #change the list tu small and then read the x dimention from it
            dY = (pts[-10])[1] - (pts[0])[1]
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
            if direction :
                dir = direction
            else :
                dir = ""
        return dir 
    
    def emp(self):
        pts = self.pts
        dX = 10
        dY = 10
#        k = 0
#        y = 0
        ok = True
        if len(pts) >= 5 and pts[-5] != 0 :
            dX = (pts[-5])[0] - (pts[0])[0]  #change the list tu small and then read the x dimention from it
#            k = (pts[-10])[0]
#            y = (pts[0])[0]
            dY = (pts[-5])[1] - (pts[0])[1]
            dxy = math.sqrt((pow(dX,2))+pow(dY,2))
            if (-5<dxy<5) :
                self.movement = self.movement+1
            else:
                self.movement = 0
            if self.movement == 5:
                ok = False    
        return (ok)
        
    