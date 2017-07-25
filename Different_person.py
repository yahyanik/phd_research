import numpy as np
import cv2
#import matplotlib.pyplot as plt
from collections import deque
import imutils
from KCF_tracker import tracker
import argparse
import sys

class find_person:
    
    
    def __init__ (self,a):  
        self.direction = []
        for track in a :
            self.direction.append(a[track].list())
        