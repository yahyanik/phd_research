#motion detection

import numpy as np
import cv2
#import matplotlib.pyplot as plt
from collections import deque
import imutils
import argparse
import sys

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

